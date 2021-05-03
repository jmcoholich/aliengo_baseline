import numpy as np


class Observation():
    def __init__(self, parts, quadruped):
        """Take a list of things to include in the observation
        and a AliengoQuadruped object.
        Return arbitrary observation upper and lower bound
        (values are not used) vectors of the correct length.
        """
        assert isinstance(parts, list)
        self.parts = parts
        self.quadruped = quadruped
        self.handles = {
            'joint_torques': self.get_applied_torques,
            'joint_positions': self.get_joint_positions,
            'joint_velocities': self.get_joint_velocities,
            'base_angular_velocity': self.get_base_angular_velocity,
            'base_roll': self.get_base_roll,
            'base_pitch': self.get_base_pitch,
            'base_yaw': self.get_base_yaw,
            'trajectory_generator_phase': self.get_tg_phase,
            'current_footstep_foot': self.get_current_foot,
            'next_footstep_distance': self.get_next_footstep_distance,
            'noise': self.noise,
            'constant_zero': self.zero,
            'one_joint_only': self.one_joint_only,
            'current_footstep_foot_one_hot': self.current_foot_one_hot
        }
        self.lengths = {
            'joint_torques': 12,
            'joint_positions': 12,
            'joint_velocities': 12,
            'base_angular_velocity': 3,
            'base_roll': 1,
            'base_pitch': 1,
            'base_yaw': 1,
            'trajectory_generator_phase': 2,
            'current_footstep_foot': 1,
            'next_footstep_distance': 3,
            'noise': 1,  # this is arbitrary
            'constant_zero': 1,
            'one_joint_only': 1,
            'current_footstep_foot_one_hot': 4
        }
        assert all(part in self.handles.keys() for part in parts)
        # ensure env is invariant to order of obs parts listed in config file
        self.parts.sort()

        self.obs_len = 0
        for part in parts:
            self.obs_len += self.lengths[part]

        # the below bounds are arbitrary and not used in the RL algorithms
        self.observation_lb = -np.ones(self.obs_len)  # TODO verify this
        self.observation_ub = np.ones(self.obs_len)

    def __call__(self):
        obs = np.concatenate([self.handles[part]() for part in self.parts])
        return obs

    def get_applied_torques(self):
        return self.quadruped.applied_torques

    def get_joint_positions(self):
        return self.quadruped.joint_positions

    def get_joint_velocities(self):
        return self.quadruped.joint_velocities

    def get_base_angular_velocity(self):
        return self.quadruped.base_avel

    def get_base_roll(self):
        return self.quadruped.base_euler[np.newaxis, 0]

    def get_base_pitch(self):
        return self.quadruped.base_euler[np.newaxis, 1]

    def get_base_yaw(self):
        return self.quadruped.base_euler[np.newaxis, 2]

    def get_tg_phase(self):
        return np.array([np.sin(self.quadruped.phases[0]),
                         np.cos(self.quadruped.phases[0])])

    def get_current_foot(self):
        return np.array([
            self.quadruped.footstep_generator.current_footstep % 4])

    def get_next_footstep_distance(self):
        return self.quadruped.footstep_generator.get_current_footstep_distance()

    def noise(self):
        return (np.random.random_sample(1) - 0.5) * 2.0

    def zero(self):
        return np.zeros(1)

    def one_joint_only(self):
        return self.quadruped.joint_positions[np.newaxis, 2]

    def current_foot_one_hot(self):
        foot = self.quadruped.footstep_generator.current_footstep % 4
        one_hot = np.zeros(4)
        one_hot[foot] = 1.0
        return one_hot
