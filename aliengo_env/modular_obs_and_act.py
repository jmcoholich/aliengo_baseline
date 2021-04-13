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
        self.quadruped = quadruped
        self.handles = {'joint_torques': self.get_applied_torques, 
                'joint_positions': self.get_joint_positions, 
                'joint_velocities': self.get_joint_velocities, 
                'base_angular_velocity': self.get_base_angular_velocity,
                'base_roll': self.get_base_roll,
                'base_pitch': self.get_base_pitch,
                'base_yaw': self.get_base_yaw,
                'trajectory_generator_phases': self.get_tg_phases}
        self.lengths = {'joint_torques': 12, 
                'joint_positions': 12, 
                'joint_velocities': 12, 
                'base_angular_velocity': 3,
                'base_roll': 1,
                'base_pitch': 1,
                'base_yaw': 1}
                # 'trajectory_generator_phases': 8}
        assert all(part in self.handles.keys() for part in parts)
        self.parts.sort() # to insure that simply passing the parts in a different order doesn't specify a different env

        self.obs_len = 0
        for part in parts: self.obs_len += self.lengths[part] 
        self.observation_lb = -np.ones(self.obs_len)
        self.observation_ub = np.ones(self.obs_len)
    

    def get_applied_torques(self):
        return self.quadruped.applied_torques

    def get_joint_positions(self):
        return self.quadruped.joint_positions

    def get_joint_velocities(self):
        return self.quadruped.joint_velocities

    def get_base_angular_velocity(self):
        return self.quadruped.base_avel

    def get_base_roll(self):
        return self.quadruped.base_orientation[0, None]

    def get_base_pitch(self):
        return self.quadruped.base_orientation[1, None]

    def get_base_yaw(self):
        return self.quadruped.base_orientation[2, None]

    def get_tg_phases(self):
        return np.concatenate((np.sin(self.quadruped.phases), np.cos(self.quadruped.phases)))

    def __call__(self):
        """Gets observation and returns it. """
        # obs = np.zeros(self.obs_len)
        obs = np.concatenate([self.handles[part]() for part in self.parts])
        return obs


class Action():
    def __init__(self, action_space, quadruped):
        """Parts is a dict where the keys are parts of the action space, and the values are dicts with keys for upper bound
        and lower bound. """

        assert isinstance(action_space, dict) # {name_of_space: dict of parameters}
        assert len(action_space) == 1
        self.action_space = action_space
        self.quadruped = quadruped
        self.allowed = {'Iscen_PMTG': self.quadruped.iscen_pmtg, 
                        'joint_positions': self.quadruped.set_joint_position_targets}
        self.action_lengths = {'Iscen_PMTG': 15, 
                                'joint_positions': 12}
        assert list(self.action_space)[0] in self.allowed.keys()
        self.action_function = self.allowed[list(self.action_space)[0]]

        self.action_lb = -np.ones(self.action_lengths[list(self.action_space)[0]]) 
        self.action_ub = np.ones(self.action_lengths[list(self.action_space)[0]])


    def __call__(self, action, time):
        self.action_function(action, time=time, params=list(self.action_space.values())[0])
        
