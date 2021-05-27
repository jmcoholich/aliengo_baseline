"""
Implements Augmented Random Search as described in this paper:

https://arxiv.org/pdf/1803.07055.pdf

This is for continuous control space only
"""
import copy

import numpy as np
import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
# define hyperparms here for now

LR = 3e-4
N_DIRS = 10
STD = 1.0
N_TOP_DIRS = 8
N_ITERS = 100
EVAL_INT = 1
EVAL_RUNS = 4


def eval_policy(env, policy, mean_std):
    rews = np.zeros(EVAL_RUNS)
    for i in range(EVAL_RUNS):
        rews[i] = run_episode(env, mean_std, policy)

    print('Avg rew is {}'.format(rews.mean()))

def update_policy(policy, deltas, rewards):
    sort_idcs = np.argsort(rewards.max(axis=1), axis=0)
    deltas = deltas[sort_idcs]
    rewards = rewards[sort_idcs]

    rew_diff = rewards[:N_TOP_DIRS, 1] - rewards[:N_TOP_DIRS, 0]
    update = (np.expand_dims(rew_diff, (1, 2))
              * deltas[:N_TOP_DIRS]).mean(axis=0)

    norm_lr = LR/rewards[:N_TOP_DIRS].std()
    policy += norm_lr * update
    return policy

def run_episode(env, old_mean_std, policy, mean_std=None):
    obs = env.reset()
    done = False
    total_rew = 0
    while not done:
        action = policy @ (obs - old_mean_std.mean) / np.sqrt(old_mean_std.var)
        obs, rew, done, _ = env.step(action)
        if mean_std is not None:
            mean_std.update(np.expand_dims(obs, 0))  # TODO make sure this is updating in main() scope
        total_rew += rew
    return total_rew


def main():
    env = gym.make("Pendulum-v0")
    assert isinstance(env.action_space, gym.spaces.box.Box)
    assert len(env.observation_space.shape) == 1

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    policy = np.zeros((act_size, obs_size))
    mean_std = RunningMeanStd()  # set epsilon to zero?

    for i in range(N_ITERS):
        old_mean_std = copy.deepcopy(mean_std)
        deltas = np.zeros((N_DIRS, *policy.shape))
        rewards = np.zeros((N_DIRS, 2))
        for j in range(N_DIRS):
            # generate and evaluate perturbations
            deltas[j] = np.random.random_sample(policy.shape)
            rewards[j, 0] = run_episode(
                env,
                old_mean_std,
                policy - deltas[j],
                mean_std=mean_std)
            rewards[j, 1] = run_episode(
                env,
                old_mean_std, policy + deltas[j],
                mean_std=mean_std)

        policy = update_policy(policy, deltas, rewards)
        if i % EVAL_INT == 0:
            eval_policy(env, policy, copy.deepcopy(mean_std))


if __name__ == "__main__":
    main()
