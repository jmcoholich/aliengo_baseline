import time
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from gym.wrappers.clip_action import ClipAction
import gym
from aliengo_env.env import AliengoEnv
import wandb


def update_mean_std(pool_output, mean_std):
    for j in range(len(pool_output)):
        mean_std.update_from_moments(pool_output[j][3],
                                     pool_output[j][4],
                                     pool_output[j][1])


def parallel_runs(policy, n_dirs, deltas, mean_std, pool, delta_std):
    pool_policies = np.tile(policy, (n_dirs * 2, 1, 1))
    pool_policies[::2] -= deltas * delta_std
    pool_policies[1::2] += deltas * delta_std
    pool_inputs = [(mean_std.mean, mean_std.var, pool_policies[k])
                   for k in range(n_dirs * 2)]
    return pool.starmap(run_episode, pool_inputs)


def create_env(env_name, env_params, seed):
    if env_name == "aliengo":
        env_ = AliengoEnv(**env_params)
        env_ = ClipAction(env_)
    else:
        env_ = gym.make(env_name)
    env_.seed(seed)
    return env_


def mp_create_env(env_name, env_params, seed):
    global env
    seed += mp.current_process()._identity[0]
    env = create_env(env_name, env_params, seed)


def eval_policy(pool, policy, mean_std, runs, total_samples, start_time):
    run_args = [(mean_std.mean, mean_std.var, policy) for _ in range(runs)]
    pool_output = pool.starmap(run_episode, run_args)

    avg_rew = 0
    avg_len = 0
    mean_info = pool_output[0][2]
    mean_info.pop('TimeLimit.truncated', None)
    for i in range(len(pool_output)):
        avg_rew += (pool_output[i][0] - avg_rew) / (i + 1)
        avg_len += (pool_output[i][1] - avg_len) / (i + 1)
        for key in list(mean_info):
            mean_info[key] += (pool_output[i][2][key] - mean_info[key]) / (i+1)

    mean_info.update({"mean_reward": float(avg_rew),
                      "num_env_samples": total_samples,
                      "hours_wall_time": (time.time() - start_time)/3600,
                      "mean_episode_length": float(avg_len)})
    wandb.log(mean_info)


def update_policy(policy, deltas, rewards, lr, top_dirs):
    sort_idcs = np.argsort(rewards.max(axis=1), axis=0)[::-1]
    deltas = deltas[sort_idcs]
    rewards = rewards[sort_idcs]

    rew_diff = rewards[:top_dirs, 1] - rewards[:top_dirs, 0]
    update = (np.expand_dims(rew_diff, (1, 2))
              * deltas[:top_dirs]).mean(axis=0)

    norm_lr = lr/(rewards[:top_dirs].std() + 1e-5)
    policy += norm_lr * update
    return policy


def run_episode(mean, var, policy):
    # keep running mean and std of the episode, then return the running mean std and the count
    obs = env.reset()
    # print("initial obs: {}".format(obs[-1]))
    # time.sleep(100)
    eps_mean_std = RunningMeanStd(shape=obs.shape)
    done = False
    total_rew = 0
    samples = 0
    while not done:
        norm_obs = (obs - mean) / np.sqrt(var)
        action = policy @ np.expand_dims(norm_obs, 1)
        obs, rew, done, info = env.step(action.flatten())
        eps_mean_std.update(np.expand_dims(obs, 0))
        total_rew += rew
        samples += 1
    return total_rew, samples, info, eps_mean_std.mean, eps_mean_std.var
