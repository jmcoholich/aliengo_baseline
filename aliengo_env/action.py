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
        self.allowed = {
            'Iscen_PMTG': self.quadruped.iscen_pmtg,
            'joint_positions': self.set_joint_position_targets,
            'one_leg_only': self.one_leg,
            'foot_positions': self.foot_positions
        }
        self.action_lengths = {
            'Iscen_PMTG': 15,
            'joint_positions': 12,
            'one_leg_only': 1,
            'foot_positions': 12
        }
        assert self.action_space in self.allowed.keys()
        self.action_function = self.allowed[self.action_space]

        self.action_lb = -np.ones(self.action_lengths[self.action_space])
        self.action_ub = np.ones(self.action_lengths[self.action_space])

    def __call__(self, action, time):
        self.action_function(action, time=time, params=self.action_space_params)

    def one_leg(self, action, time, params):
        # mildly inefficient, but this env is just for debugging anyways
        positions = np.array([0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148, 0.048225, 0.690008,
                              -1.254787, -0.050525, 0.661355, -1.243304])
        positions = self.quadruped.positions_to_actions(positions)
        positions[2] = action.item()
        self.quadruped.set_joint_position_targets(positions)

    def set_joint_position_targets(self, action, time, params):
        self.quadruped.set_joint_position_targets(action)

    def foot_positions(self, action, time, params):
        """Send true foot positions to quadruped."""
        ub = np.array(params['ub'])
        lb = np.array(params['lb'])
        mean = (ub + lb) / 2.0
        range_ = (ub - lb)
        action = action.reshape((4, 3)) * range_ / 2.0
        action = action + mean
        self.quadruped.set_foot_positions(action)

    # def act_to_foot_pos(self, action):
    #     """Send true foot positions to quadruped."""
    #     lb = np.array([-0.5, -0.3, -0.53])
    #     ub = np.array([0.5, 0.3, -0.15])
    #     mean = (ub + lb) / 2.0
    #     range_ = (ub - lb)
    #     action = action.reshape((4, 3)) * range_ / 2.0
    #     action = action + mean
    #     return action.flatten()

    # def foot_pos_to_act(self, positions):
    #     """Converts positions to actions.
    #     This is just for utility/debugging."""
    #     lb = np.array([-0.5, -0.3, -0.53])
    #     ub = np.array([0.5, 0.3, -0.15])
    #     mean = (ub + lb) / 2.0
    #     range_ = (ub - lb)
    #     action = positions.reshape((4, 3)) - mean
    #     action = action * 2.0/range_
    #     return action.flatten()
