import numpy as np


class Action():
    def __init__(self, action_space, quadruped):
        """Action_space is a dict where the key is the name of the action space, and the values are dicts with keys for
        upper bound and lower bound. """

        assert isinstance(action_space, dict)  # {name_of_space: dict of parameters}
        assert len(action_space) == 1
        self.action_space = list(action_space)[0]
        self.action_space_params = action_space[self.action_space]
        self.quadruped = quadruped
        """The keys should be function handles that the actions and time are be passed directly to as the only arg
        So far, there are no actions which have parameters other than bounds"""
        self.allowed = {'Iscen_PMTG': self.quadruped.iscen_pmtg,
                        'joint_positions': self.quadruped.set_joint_position_targets,  # TODO fix this!!
                        'one_leg_only': self.one_leg}
        self.action_lengths = {'Iscen_PMTG': 15,
                               'joint_positions': 12,
                               'one_leg_only': 1}
        assert self.action_space in self.allowed.keys()
        self.action_function = self.allowed[self.action_space]

        # the default action bounds are plus/minus 1 for all actions -- by design
        if self.action_space_params['lb'] is None:
            self.action_lb = -np.ones(self.action_lengths[self.action_space])
        else:
            self.action_lb = self.action_space_params['lb']

        if self.action_space_params['ub'] is None:
            self.action_ub = np.ones(self.action_lengths[self.action_space])
        else:
            self.action_ub = self.action_space_params['ub']

    def __call__(self, action, time):
        self.action_function(action, time=time, params=self.action_space_params)

    def one_leg(self, action, time, params):
        # mildly inefficient, but this env is just for debugging anyways
        positions = np.array([0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148, 0.048225, 0.690008,
                              -1.254787, -0.050525, 0.661355, -1.243304])
        positions = self.quadruped.positions_to_actions(positions)
        positions[2] = action.item()
        self.quadruped.set_joint_position_targets(positions)
