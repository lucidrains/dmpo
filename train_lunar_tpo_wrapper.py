# /// script
# dependencies = [
#     "dmpo",
#     "fire",
#     "gymnasium[box2d]",
#     "gymnasium[other]",
#     "tqdm",
#     "x-mlps-pytorch>=0.2.0"
# ]
# ///


import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import shutil
from functools import partial

import fire
import gymnasium as gym

from dmpo import TPO
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

def divisible_by(x, n):
    return (x % n) == 0

def main(
    cpu: bool = False,
    num_iterations: int = 2000,
    group_size: int = 64,
    epochs: int = 4,
    lr: float = 3e-4,
    eta: float = 1.0,
    entropy_coef: float = 0.01,
    divergence: str = 'forward_kl',
    record_every: int = 5,
    continuous: bool = False,
    max_episode_steps: int = 500
):
    # discrete: 4 actions -> 4 logits
    # continuous: 2 actions -> 4 logits (mean + log_var per action)

    num_actions = 2 if continuous else 4
    dim_out = num_actions * 2 if continuous else num_actions

    actor = ResidualNormedMLP(dim_in = 8, dim = 24, depth = 4, dim_out = dim_out)

    mode = 'continuous' if continuous else 'discrete'
    video_folder = f'./lunar-recordings-tpo-wrapper-{mode}'
    shutil.rmtree(video_folder, ignore_errors = True)

    env = gym.make('LunarLander-v3', continuous = continuous, render_mode = 'rgb_array')
    env = gym.wrappers.RecordVideo(
        env,
        video_folder = video_folder,
        episode_trigger = partial(divisible_by, n = record_every * group_size),
        disable_logger = True
    )

    tpo_kwargs = dict(
        action_num_continuous = num_actions
    ) if continuous else dict(
        action_num_discrete = num_actions
    )

    tpo = TPO(
        actor,
        environment = env,
        max_episode_steps = max_episode_steps,
        epochs = epochs,
        group_size = group_size,
        lr = lr,
        eta = eta,
        entropy_coef = entropy_coef,
        divergence = divergence,
        cpu = cpu,
        on_result = lambda reward, pbar: pbar.set_postfix(reward = f'{reward:.2f}'),
        **tpo_kwargs
    )

    tpo(num_iterations = num_iterations)

if __name__ == '__main__':
    fire.Fire(main)
