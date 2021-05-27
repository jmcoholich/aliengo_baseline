"""
Implements Augmented Random Search as described in this paper:

https://arxiv.org/pdf/1803.07055.pdf

This is for continuous control space only
"""

import numpy as np
import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
# define hyperparms here for now

LR = 3e-4
N_DIRS = 10
STD = 1.0
N_TOP_DIRS = 5
N_ITERS = 10


def run_episode(env, mean_std, old_mean_std, policy):
    obs = env.reset()



def main():
    env = gym.make("Pendulum-v0")
    assert isinstance(env.action_space, gym.spaces.box.Box)
    assert len(env.observation_space.shape[0]) == 0

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    policy = np.zeros((act_size, obs_size))
    mean_std = RunningMeanStd()  # set epsilon to zero?

    for _ in range(N_ITERS):
        old_mean_std = mean_std.copy()
        for _ in range(N_DIRS):
            delta = np.random.random_sample(policy.shape)
            reward_plus = run_episode(env, mean_std, old_mean_std, policy + delta)
            reward_minus = run_episode(env, mean_std, old_mean_std, policy - delta)




if __name__=="__main__":
    main()
