"""
Implements Augmented Random Search as described in this paper:

https://arxiv.org/pdf/1803.07055.pdf

This is for continuous control space only
"""
import argparse
import copy
import time
import os

import numpy as np
import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
import wandb
from ars_utils import update_policy, run_episode, eval_policy
from aliengo_env.env import AliengoEnv
from gym.wrappers.clip_action import ClipAction


def ars(args, config_yaml_file, seed):
    start_time = time.time()
    wandb.init(project=args.wandb_project, config=args)
    # env = gym.make("Pendulum-v0")
    if args.env_name == "aliengo":
        env = AliengoEnv(**args.env_params)
        env = ClipAction(env)
    else:
        env = gym.make(args.env_name)
        env.seed(seed)
    np.random.seed(seed)
    assert isinstance(env.action_space, gym.spaces.box.Box)
    assert len(env.observation_space.shape) == 1

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    policy = np.zeros((act_size, obs_size))
    mean_std = RunningMeanStd(shape=obs_size)  # set epsilon to zero?

    total_samples = 0
    i = 0
    eval_policy(env, policy, mean_std, args.eval_runs, total_samples,
                start_time)
    save_path = os.path.join("./trained_models", config_yaml_file + str(seed))
    while total_samples < args.n_samples:
        old_mean_std = copy.deepcopy(mean_std)
        deltas = np.random.normal(size=(args.n_dirs, *policy.shape))
        rewards = np.zeros((args.n_dirs, 2))
        for j in range(args.n_dirs):
            rewards[j, 0], samples, _ = run_episode(
                env,
                old_mean_std,
                policy - deltas[j] * args.delta_std,
                mean_std=mean_std)
            total_samples += samples
            rewards[j, 1], samples, _ = run_episode(
                env,
                old_mean_std, policy + deltas[j] * args.delta_std,
                mean_std=mean_std)
            total_samples += samples
        policy = update_policy(policy, deltas, rewards, args.lr, args.top_dirs)
        i += 1
        if i % args.eval_int == 0:
            eval_policy(env, policy, old_mean_std, args.eval_runs,
                        total_samples, start_time)
        if i % args.save_int == 0:
            np.savez(save_path, policy, old_mean_std.mean, old_mean_std.var)
    np.savez(save_path, policy, old_mean_std.mean, old_mean_std.var)


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
    ars(args, None, args.seed)


if __name__ == "__main__":
    main()
