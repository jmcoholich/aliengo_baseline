"""
Implements Augmented Random Search as described in this paper:

https://arxiv.org/pdf/1803.07055.pdf

This is for continuous control space only
"""
import argparse
import copy
import time

import numpy as np
import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
import wandb
from utils import update_policy, run_episode, eval_policy


def ars(args):
    start_time = time.time()
    wandb.init(project=args.wandb_project, config=args)
    # env = gym.make("Pendulum-v0")
    env = gym.make(args.env_name)
    env.seed(args.seed)
    np.random.seed(args.seed)
    assert isinstance(env.action_space, gym.spaces.box.Box)
    assert len(env.observation_space.shape) == 1

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    policy = np.zeros((act_size, obs_size))
    mean_std = RunningMeanStd(shape=obs_size)  # set epsilon to zero?

    total_samples = 0
    i = 0
    rew = eval_policy(env, policy, mean_std, args.eval_runs)
    wandb.log({"reward": rew,
               "samples": total_samples,
               "time (min)": (time.time() - start_time)/60})
    while total_samples < args.n_samples:
        old_mean_std = copy.deepcopy(mean_std)
        deltas = np.zeros((args.n_dirs, *policy.shape))
        rewards = np.zeros((args.n_dirs, 2))
        for j in range(args.n_dirs):
            # generate and evaluate perturbations
            deltas[j] = np.random.normal(size=policy.shape)
            rewards[j, 0], samples = run_episode(
                env,
                old_mean_std,
                policy - deltas[j] * args.delta_std,
                mean_std=mean_std)
            total_samples += samples
            rewards[j, 1], samples = run_episode(
                env,
                old_mean_std, policy + deltas[j] * args.delta_std,
                mean_std=mean_std)
            total_samples += samples
        policy = update_policy(policy, deltas, rewards, args.lr, args.top_dirs)
        i += 1
        if i % args.eval_int == 0:
            rew = eval_policy(env, policy, old_mean_std, args.eval_runs)
            wandb.log({"reward": rew,
                       "samples": total_samples,
                       "time (min)": (time.time() - start_time)/60})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str,
                        default="LunarLanderContinuous-v2")
    parser.add_argument("--wandb-project", type=str,
                        default="ARS_Lunar_Lander")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--n-dirs", type=int, default=1)
    parser.add_argument("--delta-std", type=float, default=0.02)
    parser.add_argument("--top-dirs", type=int, default=None)
    parser.add_argument("--top-dirs-frac", type=float, default=1.00)
    parser.add_argument("--n-samples", type=float, default=2e7)
    parser.add_argument("--eval-int", type=int, default=1)
    parser.add_argument("--eval-runs", type=int, default=10)

    args = parser.parse_args()
    if args.top_dirs and args.top_dirs_frac:
        raise ValueError("Cannot pass both top_dirs and top_dirs_frac")
    if args.top_dirs is None:
        args.top_dirs = int(args.top_dirs_frac * args.n_dirs) + 1
    ars(args)


if __name__ == "__main__":
    main()
