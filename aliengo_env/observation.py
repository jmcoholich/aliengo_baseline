import numpy as np


class Observation():
    def __init__(self, parts, quadruped, action_len, env):
        """Take a list of things to include in the observation
        and a AliengoQuadruped object.
        Return arbitrary observation upper and lower bound
        (values are not used) vectors of the correct length.
        """
        assert isinstance(parts, list)
        self.parts = parts
        self.quadruped = quadruped
        self.env = env
        self.prev_obs = None
        self.action_len = action_len
        # values are function handle to call, length of observation
        self.handles = {
            'joint_torques': (self.get_applied_torques, 12),
            'joint_positions': (self.get_joint_positions, 12),
            'joint_velocities': (self.get_joint_velocities, 12),

            'base_angular_velocity': (self.get_base_angular_velocity, 3),
            'base_velocity': (self.get_base_velocity, 3),
            'base_orientation': (self.get_base_orientation, 3),  # euler angles
            'base_position': (self.get_base_position, 3),
            'base_roll': (self.get_base_roll, 1),
            'base_pitch': (self.get_base_pitch, 1),
            'base_yaw': (self.get_base_yaw, 1),

            'trajectory_generator_phase': (self.get_tg_phase, 2),

            'next_footstep_distance': (self.get_next_footstep_distance, 3),
            'current_footstep_foot_one_hot': (self.get_current_foot_one_hot, 4),
            'footstep_distance_one_hot': (self.get_footstep_distance_one_hot,
                                          12),

            'robo_frame_foot_position': (self.get_foot_position, 12),
            'robo_frame_foot_velocity': (self.get_foot_velocity, 12),

            'noise': (self.get_noise, 1),
            'constant_zero': (self.get_zero, 1),
            'one_joint_only': (self.get_one_joint_only, 1),

            'start_token': (self.get_start_token, 1),
            'previous_observation': (None, None),
            'previous_action': (None, None),

            'foot_contact_states': (self.get_foot_contact_state, 4)
        }
        assert all(part in self.handles.keys() for part in parts)
        # ensure env is invariant to order of obs parts listed in config file
        self.parts.sort()

        self.obs_len = 0
        for part in self.parts:
            if part not in ['previous_observation', 'previous_action']:
                self.obs_len += self.handles[part][1]
        if 'previous_observation' in self.parts:
            self.obs_len *= 2
        if 'previous_action' in self.parts:
            self.obs_len += self.action_len

        # the below bounds are arbitrary and not used in the RL algorithms
        self.observation_lb = -np.ones(self.obs_len)  # TODO verify this
        self.observation_ub = np.ones(self.obs_len)

    def __call__(self, prev_action):
        obs = np.concatenate([self.handles[part][0]() for part in self.parts
                              if part not in ['previous_observation',
                                              'previous_action']])

        if ('previous_observation' in self.parts
                and self.env.eps_step_counter != 0):
            obs = np.concatenate((obs, self.prev_obs))
            self.prev_obs = obs[:int(len(obs)/2)]
        elif 'previous_observation' in self.parts:
            obs = np.concatenate((obs, obs))
            self.prev_obs = obs[:int(len(obs)/2)]

        if 'previous_action' in self.parts and prev_action is not None:
            obs = np.concatenate((obs, prev_action))
        elif 'previous_action' in self.parts:
            assert self.env.eps_step_counter == 0
            obs = np.concatenate((obs, np.zeros(self.action_len)))  # TODO make an actual function that determines the best value for this
        return obs

    def get_start_token(self):
        return np.array([float(self.env.eps_step_counter == 0)])

    def get_base_velocity(self):
        return self.quadruped.base_vel

    def get_base_orientation(self):
        return self.quadruped.base_euler

    def get_base_position(self):
        return self.quadruped.base_position

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

    def get_next_footstep_distance(self):
        """The footstep_generator.get_current_footstep_distance()
        returns the xyz vector aligned with the global coordinate system.
        The robot does not know its own yaw, so the vector is transformed
        to align with robot front direction.
        """
        yaw = self.quadruped.base_euler[2]
        vec = self.quadruped.footstep_generator.get_current_footstep_distance()
        # rotate by negative yaw angle to get vectors in robot frame
        rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
                            [np.sin(-yaw), np.cos(-yaw), 0.0],
                            [0.0, 0.0, 1.0]])
        return (rot_mat @ np.expand_dims(vec, 1)).squeeze()

    def get_footstep_distance_one_hot(self):
        output = np.zeros(12)
        foot = self.quadruped.footstep_generator.footstep_idcs[
            self.quadruped.footstep_generator.current_footstep % 4]
        vec = self.get_next_footstep_distance()
        output[foot*3 : (foot+1)*3] = vec
        return output

    def get_noise(self):
        return (np.random.random_sample(1) - 0.5) * 2.0

    def get_zero(self):
        return np.zeros(1)

    def get_one_joint_only(self):
        return self.quadruped.joint_positions[np.newaxis, 2]

    def get_current_foot_one_hot(self):
        foot = self.quadruped.footstep_generator.footstep_idcs[
            self.quadruped.footstep_generator.current_footstep % 4]
        one_hot = np.zeros(4)
        one_hot[foot] = 1.0
        return one_hot

    def get_foot_position(self):
        return self.quadruped.get_foot_frame_foot_positions().flatten()

    def get_foot_velocity(self):
        return self.quadruped.get_foot_frame_foot_velocities().flatten()

    def get_foot_contact_state(self):
        return self.quadruped.get_foot_contact_states()
