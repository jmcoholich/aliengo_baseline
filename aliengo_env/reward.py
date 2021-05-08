import numpy as np


class RewardFunction():
    def __init__(self, client, reward_parts, quadruped, action_size):
        assert isinstance(reward_parts, dict)

        self.client = client
        self.reward_parts = reward_parts
        self.quadruped = quadruped
        self.action_size = action_size

        self.all_terms = {
            'joint_torques_sq': self.joint_torques_sq,
            'forward_velocity': self.fwd_vel,
            'velocity_towards_footstep': self.velocity_towards_footstep,
            'footstep_reached': self.footstep_reached,
            'existance': self.existance,
            'smoothness_sq': self.smoothness_sq,
            'orientation': self.orientation,
            'lift_feet': self.lift_feet
        }
        assert all(part in self.all_terms.keys()
                   for part in self.reward_parts.keys())
        # action and prev_action are only for calculating smoothness rew
        self.action = None
        self.prev_action = None

    def __call__(self, action):
        # action argument is only for calculating smoothness reward
        self.action = action
        total_rew = 0
        rew_dict = {}
        for part in self.reward_parts.keys():
            term, raw_value = self.all_terms[part](*self.reward_parts[part])
            total_rew += term
            rew_dict[part] = raw_value
        return total_rew, rew_dict

    def joint_torques_sq(self, k):
        term = (self.quadruped.applied_torques
                * self.quadruped.applied_torques).sum()
        return k * term, term

    def fwd_vel(self, k, lb, ub):
        term = np.clip(self.quadruped.base_vel[0], lb, ub)
        return k * term, term

    def footstep_reached(self, k, distance_threshold):
        term = self.quadruped.footstep_generator.footstep_reached(
            distance_threshold)
        return k * term, term

    def velocity_towards_footstep(self, k, min_, max_):
        term = np.clip(
            self.quadruped.footstep_generator.velocity_towards_footstep(),
            min_,
            max_)
        return k * term, term

    def existance(self, k):
        return k, 1.0

    def smoothness_sq(self, k):
        """The maximum reward is zero, and the minimum reward is -1.0."""
        if self.prev_action is not None:
            diff = self.action - self.prev_action
            term = -(diff * diff).sum() / self.action_size / 2.0
        else:
            term = 0.0
        self.prev_action = self.action
        return k * term, term

    def orientation(self, k, x, y, z):
        coeffs = np.array([x, y, z])
        term = (abs(self.quadruped.base_euler) * coeffs).sum()
        return k * term, term

    def lift_feet(self, k, height):
        # if feet are in the middle 50% of the lifing half of the pmtg phase
        # TODO add a check that we are doing iscen pmtg as action space
        # TODO make this work not just for flatground
        swing_feet = ((1.25 * np.pi < self.quadruped.phases)
                      & (self.quadruped.phases < 1.75 * np.pi)).nonzero()[0]
        if len(swing_feet) == 0:
            term = 0.0
        else:
            global_pos = self.quadruped.get_global_foot_positions()
            above = len((global_pos[swing_feet, 2] > height).nonzero()[0])
            term = above / len(swing_feet)
        return k * term, term
