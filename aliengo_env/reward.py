import numpy as np


class RewardFunction():
    def __init__(self, client, reward_parts, quadruped):
        assert isinstance(reward_parts, dict)

        self.client = client
        self.reward_parts = reward_parts
        self.quadruped = quadruped

        self.all_terms = {
            'joint_torques_sq': self.joint_torques_sq,
            'forward_velocity': self.fwd_vel,
            'velocity_towards_footstep': self.velocity_towards_footstep,
            'footstep_reached': self.footstep_reached,
            'existance': self.existance
        }
        assert all(part in self.all_terms.keys()
                   for part in self.reward_parts.keys())

    def __call__(self):
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

    def velocity_towards_footstep(self, k, max_rewarded_vel):
        term = min(
            self.quadruped.footstep_generator.velocity_towards_footstep(),
            max_rewarded_vel)
        return k * term, term

    def existance(self, k):
        return k, 1.0
