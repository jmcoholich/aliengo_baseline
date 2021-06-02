import time
import multiprocessing

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


def parallel_runs(policy, n_dirs, deltas, env, mean_std, pool):
    pool_policies = np.tile(policy, (n_dirs * 2, 1, 1))
    pool_policies[::2] -= deltas
    pool_policies[1::2] += deltas
    pool_inputs = [(env, mean_std, pool_policies[k])
                   for k in range(n_dirs * 2)]
    return pool.starmap(run_episode, pool_inputs)


def create_env(env_name, env_params, seed):
    if env_name == "aliengo":
        env = AliengoEnv(**env_params)
        env = ClipAction(env)
    else:
        env = gym.make(env_name)
        env.seed(seed)
    return env


def eval_policy(env, policy, old_mean_std, runs, total_samples, start_time):
    rews = np.zeros(runs)
    lengths = np.zeros(runs)
    for i in range(runs):
        rews[i], lengths[i], info, _, _ = run_episode(env, old_mean_std, policy)
    avg_rew = rews.mean()
    avg_len = lengths.mean()
    # print('Avg rew is {}'.format(avg_rew))

    # log stuff
    info.pop('TimeLimit.truncated', None)
    info.update({"mean_reward": float(avg_rew),
                 "num_env_samples": total_samples,
                 "hours_wall_time": (time.time() - start_time)/3600,
                 "mean_episode_length": float(avg_len)})
    wandb.log(info)
    return None


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


def run_episode(env, old_mean_std, policy):
    # keep running mean and std of the episode, then return the running mean std and the count
    obs = env.reset()
    eps_mean_std = RunningMeanStd(shape=obs.shape)
    done = False
    total_rew = 0
    samples = 0
    while not done:
        norm_obs = (obs - old_mean_std.mean) / np.sqrt(old_mean_std.var)
        action = policy @ np.expand_dims(norm_obs, 1)
        obs, rew, done, info = env.step(action.flatten())
        eps_mean_std.update(np.expand_dims(obs, 0))
        total_rew += rew
        samples += 1
    return total_rew, samples, info, eps_mean_std.mean, eps_mean_std.var
