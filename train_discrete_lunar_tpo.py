# /// script
# dependencies = [
#   "accelerate",
#   "discrete-continuous-embed-readout",
#   "einops",
#   "fire",
#   "gymnasium[box2d]",
#   "memmap-replay-buffer>=0.1.4",
#   "moviepy",
#   "torch",
#   "torch-einops-utils",
#   "tqdm",
#   "wandb",
#   "x-mlps-pytorch",
#   "x-transformers"
# ]
# ///

# TPO - https://arxiv.org/abs/2604.06159

import os
import shutil
import fire
import gymnasium as gym
from collections import deque
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from einops import rearrange, repeat, einsum
from torch_einops_utils import masked_mean

from dmpo.tpo import tpo_target, tpo_forward_kl_loss, tpo_reverse_kl_loss

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer

from discrete_continuous_embed_readout import Readout
from x_mlps_pytorch import MLP
from x_transformers import Decoder

import numpy as np
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

# policy mlp

class PolicyMLP(nn.Module):
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

class PolicyTransformer(nn.Module):
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

# main

def main(
    K: int = 64,
    epochs: int = 4,
    lr: float = 3e-4,
    eta: float = 1.0,
    num_iterations: int = 2_000,
    record_every_updates: int = 5,
    entropy_coef: float = 0.01,
    reverse_kl: bool = False,
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
        wandb.init(project = 'discrete-lunar-tpo', config = locals())

    # env

    video_folder = './lunar-recordings'

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)

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

    policy = policy.to(device)

    optimizer = Adam(policy.parameters(), lr = lr)

    # replay buffer

    buffer_folder = './tpo_buffer'

    if os.path.exists(buffer_folder):
        shutil.rmtree(buffer_folder)

    buffer = ReplayBuffer(
        folder = buffer_folder,
        max_episodes = K,
        max_timesteps = K * 1000,
        fields = dict(
            state = ('float', (obs_dim,)),
            action = ('int', ())
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

        # rollout

        for k in range(K):
            state, _ = env.reset()
            episode_reward = 0.
            done = False
            cache = None

            while not done:
                state_t = rearrange(torch.tensor(state, dtype = torch.float32, device = device), 'd -> 1 1 d')

                with torch.no_grad():
                    logits, cache = policy(state_t, cache = cache)
                    logits = logits[:, -1, :]
                    action = policy.to_logits.sample(logits).item()

                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                buffer.store(state = state, action = action)
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
        rewards = data['reward']
        episode_lens = data['episode_lens']

        u = z_score(rewards)

        # mask

        max_len = states.shape[1]
        seq_idx = repeat(torch.arange(max_len, device = device), 't -> k t', k = K)
        mask = seq_idx < rearrange(episode_lens, 'k -> k 1')

        episode_lens_float = mask.sum(dim = 1).clamp(min = 1.).float()

        # compute target q

        with torch.no_grad():
            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            log_scores = (action_log_probs * mask).sum(dim = 1)

            # normalize

            norm_log_scores = log_scores / episode_lens_float

            log_q = tpo_target(norm_log_scores, u, eta)

        # gradient epochs

        for epoch in range(epochs):
            optimizer.zero_grad()

            logits, _ = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            log_scores = (action_log_probs * mask).sum(dim = 1)
            norm_log_scores = log_scores / episode_lens_float

            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim = -1)
            entropy = masked_mean(entropy, mask)

            log_p = F.log_softmax(norm_log_scores, dim = -1)

            tpo_loss_fn = tpo_reverse_kl_loss if reverse_kl else tpo_forward_kl_loss
            loss = tpo_loss_fn(log_p, log_q)

            loss = loss - entropy_coef * entropy

            loss.backward()
            optimizer.step()

        if use_wandb:
            wandb.log(dict(
                iter = it,
                reward = avg_reward,
                loss = loss.item(),
                entropy = entropy.item()
            ))

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
