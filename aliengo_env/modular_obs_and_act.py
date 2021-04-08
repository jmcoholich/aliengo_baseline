import sys
import os
import time
import numpy as np

from collections import OrderedDict

"""
Contains the classes for observations and actions that allow the construction of different combinations of 
observation and action spaces, as specified in the yaml file. 
"""


class Observation():
    def __init__(self, parts, quadruped):
        """Takes a list of things to include in the observation and a AliengoQuadruped object. 
        Returns arbitrary observation upper and lower bound (values are not used) vectors of the correct length. """

        assert isinstance(parts, list)
        self.parts = parts
        self.handles = {'joint_torques': quadruped.applied_torques, 
                'joint_positions': quadruped.joint_positions, 
                'joint_velocities': quadruped.joint_velocities, 
                'base_angular_velocity': quadruped.base_avel,
                'base_roll': quadruped.base_orientation[0, None],
                'base_pitch': quadruped.base_orientation[1, None],
                'base_yaw': quadruped.base_orientation[2, None]}
        self.lengths = {'joint_torques': 12, 
                'joint_positions': 12, 
                'joint_velocities': 12, 
                'base_angular_velocity': 3,
                'base_roll': 1,
                'base_pitch': 1,
                'base_yaw': 1}
        assert all(part in self.handles.keys() for part in parts)
        self.parts.sort() # to insure that simply passing the parts in a different order doesn't specify a different env

        self.obs_len = 0
        for part in parts: self.obs_len += self.lengths[part] 
        self.observation_lb = -np.ones(self.obs_len)
        self.observation_ub = np.ones(self.obs_len)

    def __call__(self):
        """Gets observation and returns it. """
        # obs = np.zeros(self.obs_len)
        obs = np.concatenate([self.handles[part] for part in self.parts])
        return obs


class Action():
    def __init__(self, parts, quadruped):
        """Parts is a dict where the keys are parts of the action space, and the values are dicts with keys for upper bound
        and lower bound. """

        assert isinstance(parts, dict)
        self.parts = OrderedDict(parts)
        self.allowed = ['joint_positions'] #, 'pmtg_parameters'] # joint_velocities', 'joint_torques'] 
        assert all(part in self.allowed for part in self.parts.keys())
        self.parts = sorted(self.parts.items())
        self.quadruped = quadruped
        self.action_lb, self.action_ub, _, _ = self.quadruped._find_position_bounds()


    def __call__(self, action):
        self.quadruped.set_joint_position_targets(action, true_positions=True)
        
