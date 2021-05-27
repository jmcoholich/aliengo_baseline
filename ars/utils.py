import numpy as np


def eval_policy(env, policy, old_mean_std, runs):
    rews = np.zeros(runs)
    for i in range(runs):
        rews[i], _ = run_episode(env, old_mean_std, policy)
    avg_rew = rews.mean()
    # print('Avg rew is {}'.format(avg_rew))
    return float(avg_rew)  # convert from np.float64 to just float for wandb


def update_policy(policy, deltas, rewards, lr, top_dirs):
    sort_idcs = np.argsort(rewards.max(axis=1), axis=0)[::-1]
    deltas = deltas[sort_idcs]
    rewards = rewards[sort_idcs]

    rew_diff = rewards[:top_dirs, 1] - rewards[:top_dirs, 0]
    update = (np.expand_dims(rew_diff, (1, 2))
              * deltas[:top_dirs]).mean(axis=0)

    norm_lr = lr/rewards[:top_dirs].std()
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
        obs, rew, done, _ = env.step(action.flatten())
        if mean_std is not None:
            mean_std.update(np.expand_dims(obs, 0))
        total_rew += rew
        samples += 1
    return total_rew, samples
