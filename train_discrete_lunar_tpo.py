# /// script
# dependencies = [
#   "gymnasium[box2d]",
#   "torch",
#   "tqdm",
#   "memmap-replay-buffer>=0.1.4",
#   "einops",
#   "discrete-continuous-embed-readout",
#   "moviepy",
#   "fire",
#   "accelerate",
#   "wandb"
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

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from discrete_continuous_embed_readout import Readout
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

# policy mlp

class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.to_logits = Readout(num_discrete = act_dim, dim = hidden)

    def forward(self, x):
        h = self.net(x)
        return self.to_logits(h)

# main

def main(
    K: int = 64,
    epochs: int = 4,
    lr: float = 3e-4,
    eta: float = 1.0,
    num_iterations: int = 2_000,
    record_every_updates: int = 5,
    entropy_coef: float = 0.01,
    cpu: bool = False,
    use_wandb: bool = False
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

    policy = PolicyMLP(obs_dim, act_dim).to(device)
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

            while not done:
                state_t = rearrange(torch.tensor(state, dtype = torch.float32, device = device), 'd -> 1 d')

                with torch.no_grad():
                    logits = policy(state_t)
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
        mean_len = episode_lens_float.mean()

        # compute target q

        with torch.no_grad():
            logits = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            log_scores = (action_log_probs * mask).sum(dim = 1)

            # normalize

            norm_log_scores = (log_scores / episode_lens_float) * mean_len

            q = tpo_target(norm_log_scores, u, eta)

        # gradient epochs

        for epoch in range(epochs):
            optimizer.zero_grad()

            logits = policy(states)
            log_probs = F.log_softmax(logits, dim = -1)

            action_log_probs = log_probs.gather(2, rearrange(actions, 'k t -> k t 1'))
            action_log_probs = rearrange(action_log_probs, 'k t 1 -> k t')

            log_scores = (action_log_probs * mask).sum(dim = 1)
            norm_log_scores = (log_scores / episode_lens_float) * mean_len

            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim = -1)
            entropy = (entropy * mask).sum() / mask.sum().clamp(min = 1.)

            log_p = F.log_softmax(norm_log_scores, dim = -1)
            loss = tpo_loss(log_p, q)

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
