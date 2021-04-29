import numpy as np


class Observation():
    def __init__(self, parts, quadruped):
        """Takes a list of things to include in the observation and a AliengoQuadruped object.
        Returns arbitrary observation upper and lower bound (values are not used) vectors of the correct length. """

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
                        'trajectory_generator_phases': self.get_tg_phases,
                        'current_footstep_foot': self.get_current_foot,
                        'next_footstep_distance': self.get_next_footstep_distance,
                        'noise': self.noise
                        }
        self.lengths = {
                        'joint_torques': 12,
                        'joint_positions': 12,
                        'joint_velocities': 12,
                        'base_angular_velocity': 3,
                        'base_roll': 1,
                        'base_pitch': 1,
                        'base_yaw': 1,
                        'trajectory_generator_phases': 8,
                        'current_footstep_foot': 1,
                        'next_footstep_distance': 3,
                        'noise': 1  # this is arbitrary
                        }
        assert all(part in self.handles.keys() for part in parts)
        self.parts.sort() # to insure that simply passing the parts in a different order doesn't specify a different env

        self.obs_len = 0
        for part in parts:
            self.obs_len += self.lengths[part]

        # the below bounds are arbitrary and not used in the RL algorithms
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

    def get_current_foot(self):
        return np.array([self.quadruped.footstep_generator.current_footstep % 4])

    def get_next_footstep_distance(self):
        return self.quadruped.footstep_generator.get_current_footstep_distance()

    def noise(self):
        return (np.random.random_sample(1) - 0.5) * 2.0

    def __call__(self):
        """Gets observation and returns it. """
        # obs = np.zeros(self.obs_len)
        obs = np.concatenate([self.handles[part]() for part in self.parts])
        return obs
