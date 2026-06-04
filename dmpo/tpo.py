from __future__ import annotations

from collections import deque, namedtuple
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Adam
import torch.nn.functional as F

from einops import rearrange, einsum, reduce
from torch_einops_utils import masked_mean, lens_to_mask

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from discrete_continuous_embed_readout import ParameterlessReadout

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def z_score(t, eps = 1e-8):
    return (t - t.mean()) / (t.std(unbiased = False) + eps)

# tpo loss functions

LogScoreReturn = namedtuple('LogScoreReturn', ['log_scores', 'logits'])

def tpo_target(log_scores, u, eta = 1.0):
    return F.log_softmax(log_scores + u / eta, dim = -1)

def tpo_forward_kl_loss(log_p, log_q):
    q = log_q.exp()
    return -einsum(q, log_p, '... k, ... k -> ...').mean()

def tpo_reverse_kl_loss(log_p, log_q):
    p = log_p.exp()
    return einsum(p, log_p - log_q, '... k, ... k -> ...').mean()

def tpo_js_loss(log_p, log_q, weight = 0.5, eps = 1e-10):
    p = log_p.exp()
    q = log_q.exp()

    m = q.lerp(p, weight)
    log_m = m.clamp(min = eps).log()

    kl_p_m = einsum(p, log_p - log_m, '... k, ... k -> ...').mean()
    kl_q_m = einsum(q, log_q - log_m, '... k, ... k -> ...').mean()

    return kl_q_m.lerp(kl_p_m, weight)

TPO_LOSS_FNS = dict(
    forward_kl = tpo_forward_kl_loss,
    reverse_kl = tpo_reverse_kl_loss,
    js = tpo_js_loss
)

# environments

class GymEnvironment(Module):
    def __init__(
        self,
        env,
        readout,
        maybe_reshape_logits,
        action_fields,
        is_discrete,
        is_continuous,
        num_continuous = None,
        num_discrete_categories = None,
        num_discrete_logits = None,
        group_size = 64,
        max_timesteps = None,
        max_episode_steps = None,
        action_clip_range = None,
        oob_penalty_weight = 0.,
        buffer_folder = './tpo_buffer',
        overwrite_buffer_on_start = True
    ):
        super().__init__()
        self.env = env
        self.readout = readout
        self.max_episode_steps = max_episode_steps
        self.action_clip_range = action_clip_range
        self.oob_penalty_weight = oob_penalty_weight

        self.is_discrete = is_discrete
        self.is_continuous = is_continuous
        self.num_continuous = num_continuous
        self.maybe_reshape_logits = maybe_reshape_logits

        self.num_discrete_categories = num_discrete_categories

        if exists(num_discrete_categories):
            categories = torch.tensor(num_discrete_categories)
            self.register_buffer('categories', categories)
            self.register_buffer('divisors', torch.cat((torch.tensor([1]), categories.cumprod(dim = 0)[:-1])))

        self.group_size = group_size

        obs_dim = int(env.observation_space.shape[0])
        max_timesteps = default(max_timesteps, group_size * 1000)

        self.buffer = ReplayBuffer(
            folder = buffer_folder,
            max_episodes = group_size,
            max_timesteps = max_timesteps,
            fields = dict(
                state = ('float', (obs_dim,)),
                **action_fields
            ),
            meta_fields = dict(
                cum_reward = 'float'
            ),
            circular = False,
            overwrite = overwrite_buffer_on_start
        )

    @property
    def is_multi_discrete(self):
        return exists(self.num_discrete_categories)

    def get_discrete_env_action(self, discrete_tensor):
        if not self.is_multi_discrete:
            return discrete_tensor.item()
        return ((discrete_tensor // self.divisors.to(discrete_tensor.device)) % self.categories.to(discrete_tensor.device)).cpu().numpy()

    def action_to_env(self, action_tensor):
        oob_amount = 0.

        if self.is_discrete and not self.is_continuous:
            return self.get_discrete_env_action(action_tensor), oob_amount

        is_mixed = self.is_discrete and self.is_continuous
        continuous_tensor = action_tensor[1] if is_mixed else action_tensor
        action = continuous_tensor.cpu().numpy()

        if exists(self.action_clip_range):
            min_val, max_val = self.action_clip_range
            oob_amount = np.maximum(0., action - max_val) + np.maximum(0., min_val - action)
            oob_amount = oob_amount.sum()
            action = np.clip(action, min_val, max_val)

        if is_mixed:
            return (self.get_discrete_env_action(action_tensor[0]), action), oob_amount

        return action, oob_amount

    def forward(self, actor):
        device = next(actor.parameters()).device
        self.buffer.clear()

        for k in range(self.group_size):
            state, _ = self.env.reset()
            episode_reward = 0.
            done = False
            step = 0

            while not done:
                state_t = torch.tensor(state, dtype = torch.float32, device = device)

                with torch.no_grad():
                    logits = self.maybe_reshape_logits(actor(state_t))
                    action_tensor = self.readout.sample(logits)

                action, oob_amount = self.action_to_env(action_tensor)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if self.oob_penalty_weight > 0.:
                    reward -= oob_amount * self.oob_penalty_weight

                step += 1

                if exists(self.max_episode_steps) and step >= self.max_episode_steps:
                    truncated = True

                done = terminated or truncated

                store_kwargs = dict(state = state)

                if self.is_discrete:
                    t = action_tensor[0] if self.is_continuous else action_tensor
                    store_kwargs['action_discrete'] = t.item()

                if self.is_continuous:
                    t = action_tensor[1] if self.is_discrete else action_tensor
                    store_kwargs['action_continuous'] = t.cpu().numpy()

                self.buffer.store(**store_kwargs)

                episode_reward += reward
                state = next_state

            self.buffer.store_meta_datapoint(k, 'cum_reward', episode_reward)
            self.buffer.advance_episode()

        return self.buffer.get_all_data(device = device)

# main class

class TPO(Module):
    def __init__(
        self,
        actor,
        environment,
        *,
        action_num_discrete = None,
        action_num_continuous = None,
        buffer_folder = './tpo_buffer',
        overwrite_buffer_on_start = True,
        max_timesteps = None,
        max_episode_steps = None,
        action_clip_range = None,
        oob_penalty_weight = 0.,
        epochs = 4,
        group_size = 64,
        optim = None,
        optim_kwargs = dict(),
        lr = 3e-4,
        max_grad_norm = None,
        eta = 1.0,
        min_rewards_std = 1e-4,
        entropy_coef = 0.01,
        divergence = 'forward_kl',
        reward_moving_average_len = 20,
        accelerator: Accelerator | None = None,
        cpu = False,
        on_result = None,
        **readout_kwargs
    ):
        super().__init__()

        self.has_discrete = exists(action_num_discrete)
        self.has_continuous = exists(action_num_continuous)

        assert self.has_discrete or self.has_continuous, 'must specify at least one of action_num_discrete or action_num_continuous'

        # readout

        readout_params = dict(**readout_kwargs)

        if self.has_discrete:
            readout_params['num_discrete'] = action_num_discrete

        if self.has_continuous:
            readout_params['num_continuous'] = action_num_continuous

        self.readout = ParameterlessReadout(**readout_params)

        # derive buffer field and action conversion from config

        action_fields = dict()
        num_discrete_categories = None
        self.num_discrete_logits = None

        if self.has_discrete:
            is_multi = isinstance(action_num_discrete, (tuple, list))
            action_fields['action_discrete'] = 'int'
            num_discrete_categories = tuple(action_num_discrete) if is_multi else None
            self.num_discrete_logits = sum(action_num_discrete) if is_multi else action_num_discrete

        if self.has_continuous:
            action_fields['action_continuous'] = ('float', (action_num_continuous,))

        # setup environment

        if not callable(environment):
            self.environment = GymEnvironment(
                environment,
                readout = self.readout,
                maybe_reshape_logits = self.maybe_reshape_logits,
                action_fields = action_fields,
                is_discrete = self.has_discrete,
                is_continuous = self.has_continuous,
                num_continuous = action_num_continuous,
                num_discrete_categories = num_discrete_categories,
                num_discrete_logits = self.num_discrete_logits,
                group_size = group_size,
                max_timesteps = max_timesteps,
                max_episode_steps = max_episode_steps,
                action_clip_range = action_clip_range,
                oob_penalty_weight = oob_penalty_weight,
                buffer_folder = buffer_folder,
                overwrite_buffer_on_start = overwrite_buffer_on_start
            )
        else:
            self.environment = environment

        # store refs

        self.num_continuous = action_num_continuous

        self.actor = actor

        if not exists(accelerator):
            accelerator = Accelerator(cpu = cpu)

        self.accelerator = accelerator

        self.device = self.accelerator.device

        if exists(optim):
            self.optimizer = optim
        else:
            self.optimizer = Adam(self.actor.parameters(), lr = lr, **optim_kwargs)

        self.actor, self.readout, self.optimizer = self.accelerator.prepare(
            self.actor, self.readout, self.optimizer
        )

        self.epochs = epochs
        self.eta = eta
        self.min_rewards_std = min_rewards_std
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.reward_moving_average_len = reward_moving_average_len

        assert divergence in TPO_LOSS_FNS, f'divergence must be one of {list(TPO_LOSS_FNS.keys())}'
        self.tpo_loss_fn = TPO_LOSS_FNS[divergence]

        self.on_result = on_result

    def maybe_reshape_logits(self, logits):
        if self.has_discrete and not self.has_continuous:
            return logits

        if self.has_continuous and not self.has_discrete:
            return rearrange(logits, '... (c d) -> ... c d', c = self.num_continuous)

        discrete_logits, continuous_logits = logits.split([self.num_discrete_logits, self.num_continuous * 2], dim = -1)
        continuous_params = rearrange(continuous_logits, '... (c d) -> ... c d', c = self.num_continuous)

        return (discrete_logits, continuous_params)

    def calculate_log_scores(self, states, actions, mask, episode_lens_float):
        logits = self.maybe_reshape_logits(self.actor(states))

        neg_log_probs = self.readout.calculate_loss(
            logits,
            targets = actions,
            mask = mask,
            return_unreduced_loss = True
        )

        log_scores = reduce(-neg_log_probs, 'b ... -> b', 'sum')
        log_scores = log_scores / episode_lens_float

        return LogScoreReturn(log_scores, logits)

    def forward(
        self,
        num_iterations = 2000,
        disable_pbar = False
    ):
        device = self.device
        recent_rewards = deque(maxlen = self.reward_moving_average_len)
        pbar = tqdm(range(num_iterations), desc = 'tpo training', disable = disable_pbar or num_iterations == 1)

        for it in pbar:

            # get rollout

            data = self.environment(self.actor)

            # unpack data

            states = data['state']

            if 'action' in data:
                actions = data['action']
            elif self.has_discrete and self.has_continuous:
                actions = (data['action_discrete'], data['action_continuous'])
            elif self.has_discrete:
                actions = data['action_discrete']
            else:
                actions = data['action_continuous']

            rewards = data.get('cum_reward', data.get('reward'))
            episode_lens = data.get('episode_lens')

            # log reward

            recent_rewards.extend(rewards.tolist())

            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))

            if exists(self.on_result):
                self.on_result(avg_reward, pbar)
            else:
                pbar.set_postfix(avg_reward = f'{avg_reward:.2f}')

            # calculate baseline and mask

            if rewards.std(unbiased = False) < self.min_rewards_std:
                u = torch.zeros_like(rewards)
            else:
                u = z_score(rewards)

            mask = data.get('mask')

            if not exists(mask):
                assert exists(episode_lens), 'episode_lens must be returned by environment if mask is not provided'
                mask = lens_to_mask(episode_lens, max_len = states.shape[1])

            mask = mask.to(device)

            episode_lens_float = mask.sum(dim = 1).clamp(min = 1.).float()

            # target distribution

            with torch.no_grad():
                out = self.calculate_log_scores(states, actions, mask, episode_lens_float)
                log_q = tpo_target(out.log_scores, u, self.eta)

            # train policy

            for epoch in range(self.epochs):
                self.optimizer.zero_grad()

                out = self.calculate_log_scores(states, actions, mask, episode_lens_float)

                log_p = F.log_softmax(out.log_scores, dim = -1)

                entropy = self.readout.entropy(out.logits)
                entropy = masked_mean(entropy, mask)

                loss = self.tpo_loss_fn(log_p, log_q)
                loss = loss - self.entropy_coef * entropy

                self.accelerator.backward(loss)

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.optimizer.step()
