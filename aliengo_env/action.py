import sys
import os
import time
import numpy as np

from collections import OrderedDict

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