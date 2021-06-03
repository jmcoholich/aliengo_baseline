"""
Implements Augmented Random Search as described in this paper:

https://arxiv.org/pdf/1803.07055.pdf

This is for continuous control space only.
"""
import argparse
import time
import os
import multiprocessing as mp

import numpy as np
import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
import wandb
from ars_utils import (update_policy, eval_policy, create_env, parallel_runs,
                       update_mean_std, mp_create_env)


def ars(args, config_yaml_file, seed):
    start_time = time.time()
    wandb.init(project=args.wandb_project, config=args)
    np.random.seed(seed)

    env = create_env(args.env_name, args.env_params, seed)
    assert isinstance(env.action_space, gym.spaces.box.Box)
    assert len(env.observation_space.shape) == 1
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    del env

    pool_size = min(mp.cpu_count() - 2, max(args.eval_runs, args.n_dirs))
    pool = mp.Pool(
        processes=pool_size,
        initializer=mp_create_env,
        initargs=(args.env_name, args.env_params, seed))

    policy = np.zeros((act_size, obs_size))
    mean_std = RunningMeanStd(shape=obs_size, epsilon=0.0)

    total_samples = 0
    update_num = 0
    eval_policy(pool, policy, mean_std, args.eval_runs, total_samples,
                start_time)
    save_path = os.path.join("./trained_models", config_yaml_file + str(seed))
    rewards = np.zeros((args.n_dirs, 2))

    while total_samples < args.n_samples:
        deltas = np.random.normal(size=(args.n_dirs, *policy.shape))

        pool_output = parallel_runs(policy, args.n_dirs, deltas, mean_std,
                                    pool, args.delta_std)
        for j in range(len(pool_output)):
            rewards[j // 2, j % 2] = pool_output[j][0]
            total_samples += pool_output[j][1]

        policy = update_policy(policy, deltas, rewards, args.lr, args.top_dirs)

        update_num += 1
        if update_num % args.eval_int == 0:
            eval_policy(pool, policy, mean_std, args.eval_runs, total_samples,
                        start_time)
        if update_num % args.save_int == 0 or total_samples >= args.n_samples:
            np.savez(save_path, policy, mean_std.mean, mean_std.var)

        update_mean_std(pool_output, mean_std)

    pool.close()
    pool.join()


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
