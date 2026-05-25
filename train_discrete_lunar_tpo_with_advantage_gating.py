# /// script
# dependencies = [
#   "accelerate",
#   "assoc-scan",
#   "discrete-continuous-embed-readout",
#   "einops",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d]",
#   "memmap-replay-buffer>=0.1.4",
#   "moviepy",
#   "torch",
#   "tqdm",
#   "wandb",
#   "x-mlps-pytorch",
#   "x-transformers"
# ]
# ///

# TPO with sequence level credit assignment and advantage gating

import os
from shutil import rmtree
from collections import deque

import fire
import gymnasium as gym
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module
from torch.optim import Adam

from einops import rearrange, repeat, einsum

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from ema_pytorch import EMA
from assoc_scan import AssocScan

from discrete_continuous_embed_readout import Readout
from x_mlps_pytorch import MLP
from x_transformers import Decoder

import wandb

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def z_score(t, eps = 1e-8):
    return (t - t.mean()) / (t.std(unbiased = False) + eps)

# tpo functions

def tpo_target(log_scores, u, eta = 1.0):
    return (F.log_softmax(log_scores, dim = -1) + u / eta).softmax(dim = -1)

def tpo_loss(log_p, q):
    return -einsum(q, log_p, '... k, ... k -> ...').mean()

def calc_gae(rewards, values, masks, gamma = 0.99, lam = 0.95):
    assert values.shape[-1] == rewards.shape[-1]

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = False)
    gae = scan(gates, delta)

    return gae + values, gae

# policy mlp

class PolicyMLP(Module):
    def __init__(self, obs_dim, act_dim, hidden = 64):
        super().__init__()
        self.net = MLP(
            obs_dim,
            hidden,
            hidden,
            activation = nn.Tanh(),
            activate_last = True
        )

        self.to_logits = Readout(
            num_discrete = act_dim,
            dim = hidden
        )

    def forward(self, x, cache = None):
        h = self.net(x)
        out = self.to_logits(h)
        return out, None

# policy transformer

class PolicyTransformer(Module):
    def __init__(self, obs_dim, act_dim, dim = 64, depth = 4, heads = 4, attn_dim_head = 32):
        super().__init__()
        self.proj_in = nn.Linear(obs_dim, dim)

        self.net = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = attn_dim_head,
            polar_pos_emb = True
        )

        self.to_logits = Readout(
            num_discrete = act_dim,
            dim = dim
        )

    def forward(self, x, cache = None):
        x = self.proj_in(x)
        h, new_cache = self.net(x, cache = cache, return_hiddens = True)
        return self.to_logits(h), new_cache

# value network

class ValueMLP(Module):
    def __init__(self, obs_dim, hidden = 64):
        super().__init__()
        self.net = MLP(obs_dim, hidden, hidden, activation = nn.Tanh(), activate_last = True)
        self.to_value = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.to_value(self.net(x)).squeeze(-1)

# main

def main(
    K: int = 64,
    epochs: int = 4,
    lr: float = 1e-3,
    critic_lr: float = 1e-3,
    eta: float = 1.0,
    start_temperature: float = 20.0,
    end_temperature: float = 1.0,
    temperature_anneal_iters: int = 50,
    gamma: float = 0.99,
    lam: float = 0.95,
    ema_beta: float = 0.99,
    ema_update_every: int = 10,
    num_iterations: int = 2_000,
    record_every_updates: int = 5,
    entropy_coef: float = 0.01,
    critic_loss_weight: float = 0.5,
    cpu: bool = True,
    use_wandb: bool = True,
    policy_type: str = 'mlp',
    transformer_dim: int = 64,
    transformer_depth: int = 4,
    transformer_heads: int = 4,
    transformer_attn_dim_head: int = 32
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if use_wandb:
        wandb.init(project = 'discrete-lunar-tpo-advantage-gated', config = locals())

    # env

    video_folder = f'./lunar-recordings-temp-anneal'

    rmtree(video_folder, ignore_errors = True)

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

    env = gym.wrappers.RecordVideo(
        env,
        video_folder = video_folder,
        episode_trigger = lambda x: divisible_by(x, record_every_updates * K)
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    # model and optimizer

    if policy_type == 'transformer':
        policy = PolicyTransformer(
            obs_dim = obs_dim,
            act_dim = act_dim,
            dim = transformer_dim,
            depth = transformer_depth,
            heads = transformer_heads,
            attn_dim_head = transformer_attn_dim_head
        )
    else:
        policy = PolicyMLP(
            obs_dim = obs_dim,
            act_dim = act_dim
        )

    critic = ValueMLP(obs_dim)

    policy = policy.to(device)
    critic = critic.to(device)

    ema_critic = EMA(critic, beta = ema_beta, update_every = ema_update_every).to(device)

    optimizer = Adam([
        dict(params = policy.parameters(), lr = lr),
        dict(params = critic.parameters(), lr = critic_lr)
    ])

    # replay buffer

    buffer_folder = f'./tpo_advantage_gated_buffer_temp_anneal'

    rmtree(buffer_folder, ignore_errors = True)

    buffer = ReplayBuffer(
        folder = buffer_folder,
        max_episodes = K,
        max_timesteps = K * 1000,
        fields = dict(
            state = ('float', (obs_dim,)),
            action = ('int', ()),
            step_reward = ('float', ()),
            is_boundary = ('bool', ())
        ),
        meta_fields = dict(
            reward = 'float'
        ),
        circular = False
    )

    # training

    recent_rewards = deque(maxlen = 20)

    pbar = tqdm(range(num_iterations), desc = 'tpo training')

    for it in pbar:
        buffer.clear()

        # anneal temperature
        if it < temperature_anneal_iters:
            current_temperature = start_temperature - (start_temperature - end_temperature) * (it / temperature_anneal_iters)
        else:
            current_temperature = end_temperature

        # rollout

        for k in range(K):
            state, _ = env.reset()
            episode_reward = 0.
            done = False
            cache = None

            while not done:
                state_t = rearrange(tensor(state, dtype = torch.float32, device = device), 'd -> 1 1 d')

                with torch.no_grad():
                    logits, cache = policy(state_t, cache = cache)
                    logits = logits[:, -1, :]
                    action = policy.to_logits.sample(logits).item()

                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                step_reward = reward

                if done and not term:
                    next_state_t = rearrange(tensor(next_state, dtype = torch.float32, device = device), 'd -> 1 d')
                    with torch.no_grad():
                        step_reward += gamma * ema_critic(next_state_t).item()

                buffer.store(
                    state = state,
                    action = action,
                    step_reward = step_reward,
                    is_boundary = done
                )
                episode_reward += reward
                state = next_state

            buffer.store_meta_datapoint(k, 'reward', episode_reward)
            buffer.advance_episode()
            recent_rewards.append(episode_reward)

        avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
        pbar.set_postfix(avg_reward = f'{avg_reward:.2f}')

        # get batch

        data = buffer.get_all_data(device = device)

        states = data['state']
        actions = data['action']
        episode_rewards = data['reward']
        step_rewards = data['step_reward']
        is_boundaries = data['is_boundary']
        episode_lens = data['episode_lens']

        u = z_score(episode_rewards)

        # mask

        max_len = states.shape[1]
        seq_idx = repeat(torch.arange(max_len, device = device), 't -> k t', k = K)
        mask = seq_idx < rearrange(episode_lens, 'k -> k 1')

        episode_lens_float = mask.sum(dim = 1).clamp(min = 1.).float()
        gae_masks = (1. - is_boundaries.float()) * mask.float()

        # compute target q

        with torch.no_grad():
            values = ema_critic(states)
            returns, advantages = calc_gae(step_rewards, values, gae_masks, gamma = gamma, lam = lam)

            norm_advantages = torch.zeros_like(advantages)
            norm_advantages[mask] = z_score(advantages[mask])

            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            # advantage gating

            advantage_gate = 2 * torch.sigmoid(norm_advantages / current_temperature)
            gated_action_log_probs = action_log_probs * advantage_gate

            log_scores = (gated_action_log_probs * mask).sum(dim = 1)

            # normalize

            norm_log_scores = log_scores / episode_lens_float

            q = tpo_target(norm_log_scores, u, eta)

        # gradient epochs

        for epoch in range(epochs):
            optimizer.zero_grad()

            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            values = critic(states)
            v_loss = F.mse_loss(values[mask], returns[mask])

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            # advantage gating

            advantage_gate = 2 * torch.sigmoid(norm_advantages / current_temperature)
            gated_action_log_probs = action_log_probs * advantage_gate

            log_scores = (gated_action_log_probs * mask).sum(dim = 1)
            norm_log_scores = log_scores / episode_lens_float

            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim = -1)

            # scale entropy by advantage gate
            entropy = entropy * advantage_gate

            entropy = (entropy * mask).sum() / mask.sum().clamp(min = 1.)

            log_p = F.log_softmax(norm_log_scores, dim = -1)
            pi_loss = tpo_loss(log_p, q)

            loss = pi_loss + critic_loss_weight * v_loss - entropy_coef * entropy

            loss.backward()
            optimizer.step()
            ema_critic.update()

        if use_wandb:
            wandb.log(dict(
                iter = it,
                reward = avg_reward,
                loss = loss.item(),
                pi_loss = pi_loss.item(),
                v_loss = v_loss.item(),
                entropy = entropy.item(),
                temperature = current_temperature
            ))

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
