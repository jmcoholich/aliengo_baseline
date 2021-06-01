import time

import numpy as np
import wandb


def eval_policy(env, policy, old_mean_std, runs, total_samples, start_time):
    rews = np.zeros(runs)
    lengths = np.zeros(runs)
    for i in range(runs):
        rews[i], lengths[i], info = run_episode(env, old_mean_std, policy)
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


def run_episode(env, old_mean_std, policy, mean_std=None):
    obs = env.reset()
    done = False
    total_rew = 0
    samples = 0
    while not done:
        norm_obs = (obs - old_mean_std.mean) / np.sqrt(old_mean_std.var)
        action = policy @ np.expand_dims(norm_obs, 1)
        obs, rew, done, info = env.step(action.flatten())
        if mean_std is not None:
            mean_std.update(np.expand_dims(obs, 0))
        total_rew += rew
        samples += 1
    return total_rew, samples, info
