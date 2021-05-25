'''
Implements the class for the Aliengo robot to be used in all the environments in this repo. All inputs and outputs
should be numpy arrays.
'''

import pybullet as p
import numpy as np
import os
import time
import warnings
import sys

from footstep_generator import FootstepGenerator


class AliengoQuadruped:
    def __init__(
        self,
        pybullet_client,
        max_torque=44.4,  # from URDF
        kp=1.0,
        kd=1.0,
        vis=False,
        footstep_params=None,
        fixed=False,
        fixed_position=None,  # leaving these here in case of debugging
        fixed_orientation=(0.0, 0.0, 0.0)
    ):
        self.max_torque = max_torque
        self.kp = kp
        self.kd = kd
        self.client = pybullet_client
        self.n_motors = 12

        # FR, FL, RR, RL
        self.foot_links = [5, 9, 13, 17]
        self.shin_links = [4, 8, 12, 16]
        self.thigh_links = [3, 7, 11, 15]
        self.hip_links = [2, 6, 10, 14]
        self.quadruped = self.load_urdf(
            fixed=fixed,
            fixed_position=fixed_position,
            fixed_orientation=fixed_orientation)

        # indices are in order of [shoulder, hip, knee] for FR, FL, RR, RL.
        # The skipped numbers are fixed joints in the URDF
        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
        self.hip_joints = [2, 6, 10, 14]
        self.thigh_joints = [3, 7, 11, 15]
        self.knee_joints = [4, 8, 12, 16]
        # 19 links if the "base" link is included (massless, 0.001 cube)
        self.num_links = 18
        (self.positions_lb,
            self.positions_ub,
            self.position_mean,
            self.position_range) = self._find_position_bounds()
        self.foot_friction_coeffs = np.array([0.5] * 4)
        self.original_link_masses = np.array(
            [self.client.getDynamicsInfo(self.quadruped, i)[0]
             for i in range(self.num_links)])
        self.link_masses = []
        # this is for the visualization when debug = True for heightmap
        self._debug_ids = []

        self._init_vis = True  # TODO get rid of this

        self.num_foot_terrain_scan_points = 10  # per foot
        self.vis = vis
        if vis:
            self.init_vis()

        # state variables
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.reaction_forces = np.zeros(12)
        self.applied_torques = np.zeros(12)

        self.base_position = np.zeros(3)
        self.base_orientation = np.zeros(4)  # quaternions
        self.base_euler = np.zeros(3)  # base orientation in Euler angles
        self.base_vel = np.zeros(3)
        self.base_avel = np.zeros(3)
        self.foot_normal_forces = np.zeros(4)
        # flag to prevent multiple observation calls without a state update
        self.state_is_updated = False

        # observation variables
        # the issue is that these are not determined by self.update_state(). I could figure out self.last_foot_position
        # command by back_calculating the foot position, which ASSUMES that robot joint positions are reset at start
        # of episode. I can update phases the same way, just setting them to the phase when t = 0.
        self.phases = None  # the issue is that this is not initialized when
        self.f_i = None

        # TODO eventually implement the below for more complex agents with
        # smoothing rewards
        # self.foot_target_history = [None] * 3
        # self.joint_pos_error_history = [None] * 3 # used in Hutter PMTG observation
        # self.joint_velocity_history = [None] * 3 # used in Hutter PMTG observation
        # self.true_joint_position_target_history = [None] * 3
        # self.last_foot_disturbance = np.zeros(6) # this is a wrench
        # self.last_torso_disturbance = np.zeros(4) # this is a foot index and a force

        # this is only used for vis rn
        self.last_global_foot_target = np.zeros((4, 3))

        # # enable force/torque sensing for knee joints, to avoid walking that excessively loads them
        # for joint in self.knee_joints:
        #     self.client.enableJointForceTorqueSensor(self.quadruped, joint, enableSensor=True)

        if footstep_params is not None:
            self.footstep_generator = FootstepGenerator(footstep_params,
                                                        self,
                                                        vis=self.vis)

        self.max_joint_vel = np.ones(self.n_motors) * 40.0  # from URDF

    def init_vis(self):
        small_ball = self.client.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[0, 155, 255, 0.75])
        self.foot_scan_balls = [0] * self.num_foot_terrain_scan_points * 4
        self.foot_text = [0] * self.num_foot_terrain_scan_points * 4
        for i in range(self.num_foot_terrain_scan_points * 4):
            self.foot_scan_balls[i] = self.client.createMultiBody(
                baseVisualShapeIndex=small_ball)
            # self.foot_text[i] = self.client.addUserDebugText('init', textPosition=[0,0,0], textColorRGB=[0]*3)
        # self.client.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
        # self.client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # self.client.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0) # perhaps my GPU rendering just isn't workin
        # for some reason, since this option doesn't change performance at all.

        # for vis coordinate system
        self.line_length = 0.25
        self.line_ids = [[0] * 3] * 4
        self.line_width = 5
        # for i in range(4):
        #     self.line_ids[i][0] = self.client.addUserDebugLine([0,0,0], [self.line_length,0,0], lineColorRGB=[1,0,0],
        #                                                     parentLinkIndex=self.thigh_links[i],
        #                                                     parentObjectUniqueId=self.quadruped,
        #                                                     lineWidth=self.line_width)
        #     self.line_ids[i][1] = self.client.addUserDebugLine([0,0,0], [0,self.line_length,0], lineColorRGB=[0,1,0],
        #                                                     parentLinkIndex=self.thigh_links[i],
        #                                                     parentObjectUniqueId=self.quadruped,
        #                                                     lineWidth=self.line_width)
        #     self.line_ids[i][2] = self.client.addUserDebugLine([0,0,0], [0,0,self.line_length], lineColorRGB=[0,0,1],
        #                                                     parentLinkIndex=self.thigh_links[i],
        #                                                     parentObjectUniqueId=self.quadruped,
        #                                                     lineWidth=self.line_width)

    def update_state(self, flat_ground, fake_client=None, update_priv_info=True):  # TODO get rid of this flatground shit
        """Updates state of the quadruped. This should be called
        once per env.step() and once per env.reset().
        The state is not the same as the observation.
        """

        joint_states = self.client.getJointStates(self.quadruped,
                                                  self.motor_joint_indices)
        self.joint_positions = np.array([joint_states[i][0]
                                         for i in range(self.n_motors)])
        self.joint_velocities = np.array([joint_states[i][1]
                                          for i in range(self.n_motors)])
        self.reaction_forces = np.array([joint_states[i][2]
                                         for i in range(self.n_motors)])
        self.applied_torques = np.array([joint_states[i][3]
                                        for i in range(self.n_motors)])

        # orientation is in quaternions
        temp = self.client.getBasePositionAndOrientation(self.quadruped)
        self.base_position = np.array(temp[0])
        self.base_orientation = np.array(temp[1])
        self.base_euler = np.array(self.client.getEulerFromQuaternion(
            self.base_orientation))
        self.base_vel, self.base_avel = self.client.getBaseVelocity(
            self.quadruped)
        self.foot_normal_forces = self.get_foot_contacts()

        # # most recent value stored in index 0
        # self.joint_pos_error_history.pop()
        # self.joint_pos_error_history.insert(0, self.joint_positions - self.true_joint_position_target_history[0])
        # self.joint_velocity_history.pop()
        # self.joint_velocity_history.insert(0, self.joint_velocities)

        self.state_is_updated = True

    # def _foot_clearance_rew(self):
    #     """Calculate foot clearancance reward as fraction of feet in swing
    #     phase that are higher than the highest surrounding point of the foot
    #     scan.
    #     """

    #     num_feet_in_swing = 0.0
    #     num_clearance = 0.0
    #     feet_receiving_rew = []
    #     for i in range(4):
    #         if np.pi* 1.25 < self.phases[i] < 1.75 * np.pi:
    #             num_feet_in_swing += 1.0

    #             scan_points = self.privileged_info[i * self.num_foot_terrain_scan_points: \
    #                                                                         (i+1) * self.num_foot_terrain_scan_points]
    #             #TODO make sure these match up with the correct foot's phase
    #             extra_clearance = 0.03 # less than an inch
    #             if (scan_points < 0.0 - extra_clearance).all():
    #                 num_clearance += 1.0
    #                 feet_receiving_rew.append(i)

    #     rew = 0.0
    #     if num_feet_in_swing > 0:
    #         rew = num_clearance/num_feet_in_swing

    #     if self.vis:
    #         # turn the legs that receive the reward blue NOTE this makes the visualization very slow. Perhaps I can make
    #         # separate legs that overlay them and then move out of sight, lol.
    #         pass
    #         # for i in range(4):
    #         #     if i in feet_receiving_rew:
    #         #         self.client.changeVisualShape(self.quadruped, self.shin_links[i], rgbaColor=[0, 0, 255, 0.9])
    #         #     else:
    #         #         self.client.changeVisualShape(self.quadruped, self.shin_links[i], rgbaColor=[0, 0, 0, 0.75])
    #     return rew


    # def _wide_step_rew(self):
    #     '''Returns the negative average amount that the foot targets move outside the max_lateral_offset.
    #     Max reward is zero. '''

    #     # for reference:
    #     # foot_positions[[0,2], 1] = -lateral_offset
    #     # foot_positions[[1,3], 1] = lateral_offset
    #     rew = 0.0
    #     max_lateral_offset = 0.11
    #     mask1 = self.foot_target_history[0][[0,2], 1] < -np.ones(2) * max_lateral_offset # true if positions are bad
    #     rew += (mask1 * (self.foot_target_history[0][[0,2], 1] + np.ones(2) * max_lateral_offset)).sum()
    #     mask2 = self.foot_target_history[0][[1,3], 1] > np.ones(2) * max_lateral_offset # true if positions are bad
    #     rew += -(mask2 * (self.foot_target_history[0][[1,3], 1] - np.ones(2) * max_lateral_offset)).sum()
    #     return rew/4.0


    # def pmtg_reward(self):
    #     '''
    #     Returns the reward function specified in S4 here:
    #     https://robotics.sciencemag.org/content/robotics/suppl/2020/10/19/5.47.eabc5986.DC1/abc5986_SM.pdf
    #     - however, just reward fwd movement, no angular velocity reward bc command direction (+x) never changes
    #     - interestingly, all the reward functions are wrapped in an exponential function, so the agent gets
    #     exponentially increasing rewards as it gets better (up to a threshold)

    #     TODO structure code here and in environments such that I avoid repeated pybullet function calls. Perhaps this
    #     function can eventually return pmtg reward AND observation. (or write another function that calls this one
    #     to do that)

    #     Clipping lienar velocity to 1.8 based on:
    #      "...maximum walking speed exceeds 1.8 m/s" https://www.unitree.com/products/aliengo
    #     '''

    #     speed_treshold = 0.5 # m/s
    #     base_vel, base_avel = self.client.getBaseVelocity(self.quadruped)
    #     lin_vel_rew = np.exp(-2.0 * (base_vel[0] - speed_treshold) * (base_vel[0] - speed_treshold)) \
    #                                                                             if base_vel[0] < speed_treshold else 1.0

    #     # give reward if we are pointed the right direction
    #     _, _, yaw = self.client.getEulerFromQuaternion(self.base_orientation)
    #     angular_rew = np.exp(-1.5 * abs(yaw)) # if yaw is zero this is one.

    #     base_motion_rew = np.exp(-1.5 * (base_vel[1] * base_vel[1])) + \
    #                                         np.exp(-1.5 * (base_avel[0] * base_avel[0] + base_avel[1] * base_avel[1]))

    #     foot_clearance_rew = self._foot_clearance_rew()

    #     body_collision_rew = -(self.is_non_foot_ground_contact() + self.self_collision())

    #     target_smoothness_rew = - np.linalg.norm(self.foot_target_history[0] - 2 * self.foot_target_history[1] + \
    #                                                                                         self.foot_target_history[2])

    #     torque_rew = -np.linalg.norm(self.applied_torques, 1)

    #     # knee_force_rew = -np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()
    #     # knee_force_ratio_rew = np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[0,2,4]]]).sum() /\
    #     #                                                 np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()

    #     wide_step_rew = self._wide_step_rew()

    #     # rew_dict includes all the things I want to keep track of an average over an entire episode, to be logged
    #     # add terms of reward function
    #     rew_dict = {'lin_vel_rew': lin_vel_rew, 'base_motion_rew': base_motion_rew,
    #                     'body_collision_rew':body_collision_rew, 'target_smoothness_rew':target_smoothness_rew,
    #                     'torque_rew':torque_rew, 'angular_rew': angular_rew, 'foot_clearance_rew': foot_clearance_rew,
    #                     # 'knee_force_rew':knee_force_rew}
    #                     # 'knee_force_ratio_rew':knee_force_ratio_rew}
    #                     'wide_step_rew':wide_step_rew}
    #     # other stuff to track
    #     rew_dict['x_vel'] = self.base_vel[0]

    #     total_rew = 0.50 * lin_vel_rew + 0.05 * angular_rew + 0.10 * base_motion_rew + 1.00 * foot_clearance_rew \
    #         + 0.20 * body_collision_rew + 0.30 * target_smoothness_rew + 2e-5 * torque_rew \
    #         + 2.0 * wide_step_rew #0.1 * knee_force_ratio_rew #+ 0.001 * knee_force_rew
    #     return total_rew, rew_dict


    # def trot_in_place_reward(self):
    #     '''
    #     A copy of the self.pmtg_reward() function, just with the forward rew replaced with a rew for staying still.
    #     '''

    #     speed_treshold = 0.0 # m/s
    #     base_vel, base_avel = self.client.getBaseVelocity(self.quadruped)
    #     lin_vel_rew = np.exp(-2.0 * (base_vel[0] - speed_treshold) * (base_vel[0] - speed_treshold))

    #     # give reward if we are pointed the right direction
    #     _, _, yaw = self.client.getEulerFromQuaternion(self.base_orientation)
    #     angular_rew = np.exp(-1.5 * abs(yaw)) # if yaw is zero this is one.

    #     base_motion_rew = np.exp(-1.5 * (base_vel[1] * base_vel[1])) + \
    #                                         np.exp(-1.5 * (base_avel[0] * base_avel[0] + base_avel[1] * base_avel[1]))

    #     foot_clearance_rew = self._foot_clearance_rew()

    #     body_collision_rew = -(self.is_non_foot_ground_contact() + self.self_collision())

    #     target_smoothness_rew = - np.linalg.norm(self.foot_target_history[0] - 2 * self.foot_target_history[1] + \
    #                                                                                         self.foot_target_history[2])

    #     torque_rew = -np.linalg.norm(self.applied_torques, 1)

    #     # knee_force_rew = -np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()
    #     # knee_force_ratio_rew = np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[0,2,4]]]).sum() /\
    #     #                                                 np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()

    #     wide_step_rew = self._wide_step_rew()

    #     # rew_dict includes all the things I want to keep track of an average over an entire episode, to be logged
    #     # add terms of reward function
    #     rew_dict = {'lin_vel_rew': lin_vel_rew, 'base_motion_rew': base_motion_rew,
    #                     'body_collision_rew':body_collision_rew, 'target_smoothness_rew':target_smoothness_rew,
    #                     'torque_rew':torque_rew, 'angular_rew': angular_rew, 'foot_clearance_rew': foot_clearance_rew,
    #                     # 'knee_force_rew':knee_force_rew}
    #                     # 'knee_force_ratio_rew':knee_force_ratio_rew}
    #                     'wide_step_rew':wide_step_rew}
    #     # other stuff to track
    #     rew_dict['x_vel'] = self.base_vel[0]

    #     total_rew = 0.50 * lin_vel_rew + 0.05 * angular_rew + 0.10 * base_motion_rew + 1.00 * foot_clearance_rew \
    #             + 0.20 * body_collision_rew + 0.30 * target_smoothness_rew + 2e-5 * torque_rew \
    #             + 2.0 * wide_step_rew # 0.1 * knee_force_ratio_rew #+ 0.001 * knee_force_rew
    #     return total_rew, rew_dict

    # def pmtg_action(self, time, action):
    #     self.set_trajectory_parameters(time, action[0], action[1],
    #                                    action[2], action[3], action[4],
    #                                    action[5], action[6:])

    # def get_privileged_info(self, fake_client=None, flat_ground=False, ray_start=100):
    #     '''
    #     Priveledged info includes
    #     - terrain profile = scan of nine points in a 10 cm radius around each foot
    #         - this requires a replica of the simulation without the quadruped('fake_client')
    #     - foot contact states and forces
    #     - friction coefficients
    #     - link masses
    #     From page 8 of https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
    #     maybeTODO add if body is contacting?
    #     '''

    #     terrain_profile = self._get_foot_terrain_scan(fake_client, flat_ground=flat_ground, ray_start=ray_start)
    #     contact_forces = self.get_foot_contacts()
    #     privileged_info = np.concatenate((terrain_profile,
    #                                     contact_forces,
    #                                     self.foot_friction_coeffs,
    #                                     self.link_masses,
    #                                     self.last_torso_disturbance,
    #                                     self.last_foot_disturbance)) # verifies that this copies
    #     self.last_torso_disturbance = np.zeros(6) # NOTE these variables only work when disturbances last for 1 step
    #     self.last_foot_disturbance = np.zeros(4) # If I have multi-step disturbances, I will need to change this
    #     return privileged_info

    # def _get_foot_terrain_scan(self, fake_client=None, flat_ground=False, ray_start=100):
    #     '''Returns a flat array of relative heights of length 4 * self.num_foot_terrain_scan_points. The heights are
    #     relative to the bottom of the aliengo foot.
    #     NOTE concavity of terrain can't be determined with this scan setup.
    #     TODO see if starting the rays extremely is actually slower (ie do I care about making rays shorter when I can)'''

    #     foot_pos = np.array([i[0] for i in self.client.getLinkStates(self.quadruped, self.foot_links)])

    #     if flat_ground:
    #         # the return is a flat vector
    #         relative_z = -np.repeat(foot_pos[:,2] -0.0265, self.num_foot_terrain_scan_points)
    #         if self.vis:
    #             r = 0.1
    #             n = self.num_foot_terrain_scan_points
    #             x = np.linspace(0, 2*np.pi, n)
    #             scan_positions = (np.expand_dims(foot_pos[:, :-1], 1) + \
    #                                 np.expand_dims(np.stack((r * np.cos(x), r * np.sin(x))).swapaxes(0,1), 0)).reshape((4*n, 2))
    #             raw = np.zeros((len(scan_positions), 4, 3))
    #     else:
    #         if fake_client is None: raise ValueError('Need another client with same terrain to get heightmap from.')
    #         r = 0.1
    #         n = self.num_foot_terrain_scan_points
    #         x = np.linspace(0, 2*np.pi, n)
    #         scan_positions = (np.expand_dims(foot_pos[:, :-1], 1) + \
    #                             np.expand_dims(np.stack((r * np.cos(x), r * np.sin(x))).swapaxes(0,1), 0)).reshape((4*n, 2))
    #         ray_start_pos = np.concatenate((scan_positions, np.ones((4*n, 1)) * ray_start), axis=1)
    #         ray_end_pos = np.concatenate((scan_positions, np.ones((4*n, 1)) * -1), axis=1)
    #         raw = fake_client.rayTestBatch(rayFromPositions=ray_start_pos, rayToPositions=ray_end_pos, numThreads=0)
    #         relative_z = np.array([raw[i][3][2] - (foot_pos[j][2] - 0.0265) for j in range(4)\
    #                                                                                  for i in range(j * n, (j+1) * n)])

    #     if self.vis:
    #         for i in range(n * 4):
    #             pos = np.concatenate((scan_positions[i], [raw[i][3][2]]))
    #             self.client.resetBasePositionAndOrientation(self.foot_scan_balls[i], posObj=pos, ornObj=[0,0,0,1])
    #             # TODO for some reason, rendering this text increases rendering time by about 100x. Why???
    #             # self.client.addUserDebugText('{:.1f}'.format(relative_z[i]),
    #             #                                 textPosition=pos,
    #             #                                 replaceItemUniqueId=self.foot_text[i],
    #             #                                 textColorRGB=[0]*3)
    #     return relative_z

    def randomize_foot_friction(self, lb=0.3, ub=1.2):  # TODO enable these randomizations in the yaml files
        """Randomize the coefficient of friction of each foot,
        sampled from uniform random distribution. Returns the
        random coefficients.
        """

        self.foot_friction_coeffs = np.random.uniform(low=lb, high=ub, size=4)
        for i in range(4):
            self.client.changeDynamics(
                self.quadruped,
                self.foot_links[i],
                lateralFriction=self.foot_friction_coeffs[i])
        return self.foot_friction_coeffs

    def randomize_link_masses(self, lb=-0.05, ub=0.05):
        """Set link masses to random values, which are uniformly random
        percent increases/decreases to original link mass.
        Returns random masses. Excludes links that have zero mass.
        Should probably only be called when resetting
        environments.
        """

        assert lb > -1.0
        factors = np.random.uniform(low=lb, high=ub, size=self.num_links) + 1.0
        self.link_masses = self.original_link_masses * factors
        for i in range(self.num_links):
            self.client.changeDynamics(self.quadruped, i,
                                       mass=self.link_masses[i])
        return self.link_masses

    def trajectory_generator(self, time, amplitude, walking_height,
                             frequency, params):
        # TODO eventually make trajectory generator its own object in a
        # separate file
        assert time >= 0.0
        if params['gait'] == 'trot':
            # FR, FL, RR, RL
            phase_offsets = np.array([0, np.pi, np.pi, 0])
        elif params['gait'] == 'walk':
            phase_offsets = np.array([0.0, 1.0, 0.5, 1.5]) * np.pi

        self.phases = (phase_offsets + frequency * 2 * np.pi * time) % (2 * np.pi)

        foot_positions = np.zeros((4, 3))
        for i in range(4):
            z = (params['step_height'] * self._foot_step_ztraj(self.phases[i])
                 - walking_height)
            x = np.cos(self.phases[i]) * amplitude
            foot_positions[i] = np.array([x, 0, z])

        foot_positions[[0, 2], 1] -= params['lateral_offset']
        foot_positions[[1, 3], 1] += params['lateral_offset']
        foot_positions[:, 0] += params['x_offset']
        joint_targets = self.set_foot_positions(foot_positions,
                                                return_joint_targets=True)
        return joint_targets

    def iscen_pmtg(self, action, time, params):  # TODO correct this normalize
        """The action should consist of amplitude, walking_height,
        frequency, plus 12 joint positions residuals,
        all in the range of [-1, 1]
        """
        (amplitude, walking_height, frequency,
            true_residuals) = self._unpack_iscen_pmtg_args(action, params)
        true_joint_positions = self.trajectory_generator(
            time, amplitude, walking_height, frequency, params)
        self.set_joint_position_targets(true_joint_positions + true_residuals,
                                        true_positions=True)

    def _unpack_iscen_pmtg_args(self, action, params):
        """Check all action lower bounder are less than or equal to
        upper bounds as specified in yaml file. Then, map action
        from [-1, 1] to their true values.
        """
        assert (params['amplitude_bounds']['ub']
                >= params['amplitude_bounds']['lb'])
        assert (params['walking_height_bounds']['ub']
                >= params['walking_height_bounds']['lb'])
        assert (params['frequency_bounds']['ub']
                >= params['frequency_bounds']['lb'])
        assert (params['residual_bounds']['ub']
                >= params['residual_bounds']['lb'])

        amplitude = (action[0] * (params['amplitude_bounds']['ub']
                                  - params['amplitude_bounds']['lb']) * 0.5
                     + (params['amplitude_bounds']['lb']
                        + params['amplitude_bounds']['ub']) * 0.5)
        walking_height = (action[1] * (params['walking_height_bounds']['ub']
                                       - params['walking_height_bounds']['lb'])
                          * 0.5 + (params['walking_height_bounds']['lb']
                                   + params['walking_height_bounds']['ub'])
                          * 0.5)
        frequency = (action[2] * (params['frequency_bounds']['ub']
                                  - params['frequency_bounds']['lb']) * 0.5
                     + (params['frequency_bounds']['lb']
                        + params['frequency_bounds']['ub']) * 0.5)
        true_residuals = (action[3:] * (params['residual_bounds']['ub']
                                        - params['residual_bounds']['lb'])
                          * 0.5 * self.position_range * 0.5)
        return amplitude, walking_height, frequency, true_residuals

    # def get_pmtg_action_bounds(self):
    #     '''Returns bounds of actions to set_trajectory_parameters. I don't really know what these should be. (I don't
    #     think it matters.)'''

    #     # frequency adjustments, then foot position residuals
    #     lb = np.array([-0.00001] * 4 + [-0.2] * 12) # TODO
    #     ub = np.array([0.000001] * 4 + [0.2] * 12)
    #     return lb, ub

    # def _foot_step_xtraj(self, phase):
    #     # linearly go from -step_len to step_len. When phase is between pi and 2pi, I should be moving the foot fwd.
    #     # otherwise, move the foot backwards.
    #     # Go between [-1, 1]. use a nice sin wave
    #     if 0.0 <= phase < np.pi:

    #     elif np.pi <= phase <= 2.0 * np.pi:

    #     else:
    #         raise ValueError('phase is out of bounds')

    def _foot_step_ztraj(self, phase):
        """Takes a phase scalar and outputs a value [0, 1].
        This is according to this formula is S3 here,
        but normalized to [0, 1]:
        https://robotics.sciencemag.org/content/robotics/
        suppl/2020/10/19/5.47.eabc5986.DC1/abc5986_SM.pdf
        """
        assert 0 <= phase < 2 * np.pi, 'phase must be in [0, 2 * pi)'
        k = 2 * (phase - np.pi) / np.pi
        if -2 <= k < 0:
            return 0
        elif 0 <= k < 1:
            return -2 * k * k * k + 3 * k * k
        else:
            return 2 * k * k * k - 9 * k * k + 12 * k - 4

    def get_foot_frame_foot_velocities(self):
        """Returns velocities of shape (4, 3)."""
        # TODO compensate for distance to bottom of foot?? Velocities will
        # be slightly off
        info = self.client.getLinkStates(self.quadruped,
                                         self.foot_links,
                                         computeLinkVelocity=1)
        global_vels = np.array([info[i][6] for i in range(4)])
        _, base_o = self.client.getBasePositionAndOrientation(self.quadruped)
        _, _, yaw = self.client.getEulerFromQuaternion(base_o)
        # rotate the velocity vectors in the opposite direction of the yaw
        # to get vectors in the robot frame
        rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
                            [np.sin(-yaw), np.cos(-yaw), 0.0],
                            [0.0, 0.0, 1.0]])
        foot_frame_vels = (rot_mat @ np.expand_dims(global_vels, 2)).squeeze()
        return foot_frame_vels

    def get_global_foot_positions(self):
        """Returns an array of shape (4, 3) which gives the global
            positions of the bottom of the feet."""
        global_pos = np.array([i[0] for i in self.client.getLinkStates(
            self.quadruped, self.foot_links)])
        global_pos[:, 2] -= 0.0265
        return global_pos

    def get_foot_frame_foot_positions(self, global_pos=None):
        """Returns the position of the feet in the same frame of the
        set_foot_positions() argument. Z position is the
        bottom of the foot collision spheres. Return is of shape (4, 3).
        Inverse of _foot_frame_pos_to_global().
        """
        if global_pos is None:
            global_pos = self.get_global_foot_positions()
        base_p, base_o = self.client.getBasePositionAndOrientation(
            self.quadruped)
        _, _, yaw = self.client.getEulerFromQuaternion(base_o)

        # subtract off position of base
        output = global_pos - np.array(base_p)

        # rotate output to be in robot frame
        rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
                            [np.sin(-yaw), np.cos(-yaw), 0.0],
                            [0.0, 0.0, 1.0]])
        output = (rot_mat @ np.expand_dims(output, 2)).squeeze()

        # subtract off the hip positions
        # offsets are in robot frame
        hip_offset_from_base = np.array([self.client.getJointInfo(
            self.quadruped, self.hip_joints[i])[14] for i in range(4)])
        output -= hip_offset_from_base
        output[:, 2] -= 0.0265
        return output

    def foot_frame_pos_to_global(self, foot_frame_pos):
        """Take foot frame positions and output global coordinates.
        Inverse of get_foot_frame_foot_positions().
        """
        base_p, base_o = self.client.getBasePositionAndOrientation(
            self.quadruped)
        _, _, yaw = self.client.getEulerFromQuaternion(base_o)

        # add the hip positions, which are in robot frame
        hip_offset_from_base = np.array([self.client.getJointInfo(
            self.quadruped, self.hip_joints[i])[14] for i in range(4)])
        output = foot_frame_pos + hip_offset_from_base

        # rotate output to be in global frame
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                            [np.sin(yaw), np.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0]])
        output = (rot_mat @ np.expand_dims(output, 2)).squeeze()

        # add position of base
        output += np.array(base_p)
        return output

    def set_foot_positions(self, foot_positions, return_joint_targets=False):
        """Take a numpy array of shape (4, 3), representing relative foot xyz
        distance to the hip joint. Uses IK to calculate joint
        position targets and sets those targets.
        The Z-foot position represents the BOTTOM of the collision sphere
        """
        assert foot_positions.shape == (4, 3)
        self.foot_target_history.pop()
        self.foot_target_history.insert(0, foot_positions)

        commanded_global_foot_positions = self.foot_frame_pos_to_global(
            foot_positions)

        # TODO use analytic IK (probably faster and more accurate)
        # calculateInverseKinematics2 has a memory leak, so using the original
        joint_positions = np.zeros(12)
        for i in range(4):
            temp = self.client.calculateInverseKinematics(
                self.quadruped,
                self.foot_links[i],
                targetPosition=commanded_global_foot_positions[i],
                # maxNumIterations=1000,
                # residualThreshold=1e-10))
            )
            joint_positions[i*3 : (i + 1)*3] = np.array(temp)[i*3 : (i + 1)*3]
        # TODO go back to old way, since Erwin fixed it
        # old way
        # joint_positions = np.array(self.client.calculateInverseKinematics2(self.quadruped,
        #                                             self.foot_links,
        #                                             targetPositions=commanded_global_foot_positions))
        if return_joint_targets:
            return joint_positions
        self.set_joint_position_targets(joint_positions, true_positions=True)
        self.last_global_foot_target = commanded_global_foot_positions
        if self.vis:
            self.visualize()
        return None

    def visualize(self):
        """green spheres are commanded positions, red spheres are actual positions.
        TODOmaybe: add the foot coordinate frames visualization, and light up feet that are contacting.
        TODO: visualize external forces and torques.
        """

        commanded_global_foot_positions = self.last_global_foot_target
        hip_joint_positions = np.zeros((4, 3))  # storing these for use when debug
        for i in range(4):
            hip_offset_from_base = self.client.getJointInfo(self.quadruped, self.hip_joints[i])[14] #TODO just store this value
            base_p, base_o = self.client.getBasePositionAndOrientation(self.quadruped)
            hip_joint_positions[i], _ = np.array(self.client.multiplyTransforms(positionA=base_p,
                                                    orientationA=base_o,
                                                    positionB=hip_offset_from_base,
                                                    orientationB=[0.0, 0.0, 0.0, 1.0]))

        if self._init_vis:
            # balls are same radius as foot collision sphere
            commanded_ball = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.0265, rgbaColor=[0, 100, 0, 1.0])
            actual_ball = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.0265, rgbaColor=[255, 0, 0, 1.0])
            # for i in range(self.num_links):
            #     self.client.changeVisualShape(self.quadruped, i, rgbaColor=[0, 0, 0, 0.75])
            # visualize commanded foot positions
            self.foot_ball_ids = [0]*4
            self.hip_ball_ids = [0]*4
            for i in range(4):
                self.foot_ball_ids[i] = self.client.createMultiBody(baseVisualShapeIndex=commanded_ball,
                                                                basePosition=commanded_global_foot_positions[i])
            # visualize calculated hip positions
            for i in range(4):
                self.hip_ball_ids[i] = self.client.createMultiBody(baseVisualShapeIndex=actual_ball,
                                                                    basePosition=hip_joint_positions[i])
            self._init_vis = False
        else:
            for i in range(4):
                if commanded_global_foot_positions is not None:
                    self.client.resetBasePositionAndOrientation(self.foot_ball_ids[i],
                                                                posObj=commanded_global_foot_positions[i],
                                                                ornObj=[0,0,0,1])
                self.client.resetBasePositionAndOrientation(self.hip_ball_ids[i],
                                                            posObj=hip_joint_positions[i],
                                                            ornObj=[0,0,0,1])
                # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*self.f_i) + ' ' * 20, end='\r', flush=True)

    def render(self, mode, client):
        '''Returns RGB array of current scene if mode is 'rgb_array'.'''

        RENDER_WIDTH = 480
        RENDER_HEIGHT = 360

        # base_x_velocity = np.array(self.client.getBaseVelocity(self.quadruped)).flatten()[0]
        # torque_pen = -0.00001 * np.power(self.applied_torques, 2).mean()

        # RENDER_WIDTH = 960
        # RENDER_HEIGHT = 720

        # RENDER_WIDTH = 1920
        # RENDER_HEIGHT = 1080

        if mode == 'rgb_array':
            base_pos, _ = self.client.getBasePositionAndOrientation(self.quadruped)
            # base_pos = self.minitaur.GetBasePosition()
            view_matrix = client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2.0,
                yaw=0,
                pitch=-30.,
                roll=0,
                upAxisIndex=2)
            proj_matrix = client.computeProjectionMatrixFOV(fov=60,
                aspect=float(RENDER_WIDTH) /
                RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0)
            _, _, px, _, _ = client.getCameraImage(width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = np.array(px)
            img = img[:, :, :3]
            # img = putText(np.float32(img), 'X-velocity:' + str(base_x_velocity)[:6], (1, 60),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 'Torque Penalty Term: ' + str(torque_pen)[:8], (1, 80),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 'Total Rew: ' + str(torque_pen + base_x_velocity)[:8], (1, 100),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # foot_contacts = self._get_foot_contacts()
            # for i in range(4):
            #     if type(foot_contacts[i]) is list: # multiple contacts
            #         assert False
            #         num = np.array(foot_contacts[i]).round(2)
            #     else:
            #         num = round(foot_contacts[i], 2)
            #     img = putText(np.float32(img),
            #                 ('Foot %d contacts: ' %(i+1)) + str(num),
            #                 (200, 60 + 20 * i),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img),
            #                 'Body Contact: ' + str(self.is_non_foot_ground_contact()),
            #                 (200, 60 + 20 * 4),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img),
            #                 'Self Collision: ' + str(self.quadruped.self_collision()),
            #                 (200, 60 + 20 * 5),
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            return np.uint8(img)

        else:
            return



    def get_foot_contacts(self, object_=None):
        '''
        Returns a numpy array of shape (4,) containing the normal forces on each foot with the object given. If
        no object given, just checks with any object in self.client simulation.
        '''

        contacts = [0] * 4
        for i in range(len(self.foot_links)):
            if object_ is None:
                info = self.client.getContactPoints(bodyA=self.quadruped,
                                                    linkIndexA=self.foot_links[i])
            else:
                info = self.client.getContactPoints(bodyA=self.quadruped,
                                                    bodyB=object_,
                                                    linkIndexA=self.foot_links[i])
            if len(info) == 0: # leg does not contact ground
                contacts[i] = 0
            elif len(info) == 1: # leg has one contact with ground
                contacts[i] = info[0][9] # contact normal force
            else: # use the contact point with the max normal force when there is more than one contact on a leg
                #TODO investigate scenarios with more than one contact point and maybe do something better (mean
                # or norm of contact forces?)
                normals = [info[i][9] for i in range(len(info))]
                contacts[i] = max(normals)
                # print('Number of contacts on one foot: %d' %len(info))
                # print('Normal Forces: ', normals,'\n')
        contacts = np.array(contacts)
        if (contacts > 10_000).any():
            warnings.warn("Foot contact force of %.2f over 10,000 (maximum of observation space)" %max(contacts))
        return contacts

    def _get_heightmap(self, client, ray_start_height, base_position, heightmap_params, vis=False, vis_client=None):
        '''Debug flag enables printing of labeled coordinates and measured heights to rendered simulation.
        Uses the "fake_client" simulation instance in order to avoid measuring the robot instead of terrain
        ray_start_height should be a value that is guranteed to be above any terrain we want to measure.
        It is also where the debug text will be displayed when debug=True.'''

        length = heightmap_params['length']
        robot_position = heightmap_params['robot_position']
        grid_spacing = heightmap_params['grid_spacing']
        assert length % grid_spacing == 0
        grid_len = int(length/grid_spacing) + 1

        # debug = True
        # show_xy = False

        # if self._debug_ids != []: # remove the exiting debug items
        #     for _id in self._debug_ids:
        #         self.client.removeUserDebugItem(_id)
        #     self._debug_ids = []

        base_x = base_position[0]
        base_y = base_position[1]
        base_z = base_position[2]

        x = np.linspace(0, length, grid_len)
        y = np.linspace(-length/2.0, length/2.0, grid_len)
        coordinates = np.array(np.meshgrid(x,y))
        coordinates[0,:,:] += base_x - robot_position
        coordinates[1,:,:] += base_y
        # coordinates has shape (2, grid_len, grid_len)
        coor_list = coordinates.reshape((2, grid_len**2)).swapaxes(0, 1) # is now shape (grid_len**2,2)
        ray_start = np.append(coor_list, np.ones((grid_len**2, 1)) * ray_start_height, axis=1) #TODO check that this and in general the values are working properly
        ray_end = np.append(coor_list, np.zeros((grid_len**2, 1)) - 1, axis=1)
        raw_output = client.rayTestBatch(ray_start, ray_end, numThreads=0) # this should be the fake_client, without the quadruped
        z_heights = np.array([raw_output[i][3][2] for i in range(grid_len**2)])
        relative_z_heights = z_heights - base_z

        if vis:
            skip_n = 32 # don't plot all the heightmap points, it just takes forever
            if vis_client is None: raise ValueError('Must pass vis_client to visualize results')
            z_grid = z_heights.reshape((grid_len, grid_len))
            vis_shp = vis_client.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0., 0., 0., 1.])
            for i in range(0, grid_len, skip_n):
                for j in range(0, grid_len, skip_n):
                    vis_client.createMultiBody(baseVisualShapeIndex=vis_shp,
                                            basePosition=[coordinates[0,i,j], coordinates[1,i,j], z_grid[i,j]])

        # if debug:
        #     # #print xy coordinates of robot origin
        #     # _id = self.client.addUserDebugText(text='%.2f, %.2f'%(base_x, base_y),
        #     #             textPosition=[base_x, base_y,ray_start_height+1],
        #     #             textColorRGB=[0,0,0])
        #     # self._debug_ids.append(_id)
        #     for i in range(grid_len):
        #         for j in range(grid_len):
        #             # if show_xy:
        #             #     text = '%.3f, %.3f, %.3f'%(coordinates[0,i,j], coordinates[1,i,j], z_heights.reshape((grid_len, grid_len))[i,j])
        #             # else:
        #             #     text = '%.3f'%(z_heights.reshape((grid_len, grid_len))[i,j])
        #             # _id = self.client.addUserDebugText(text=text,
        #             #                         textPosition=[coordinates[0,i,j], coordinates[1,i,j],ray_start_height+0.5],
        #             #                         textColorRGB=[0,0,0])
        #             self._debug_ids.append(_id)
        #             _id = self.client.addUserDebugLine([coordinates[0,i,j], coordinates[1,i,j],ray_start_height+0.5],
        #                                     [coordinates[0,i,j], coordinates[1,i,j], 0],
        #                                     lineColorRGB=[0,0,0] )
        #             self._debug_ids.append(_id)

        return relative_z_heights.reshape((grid_len, grid_len))

    def is_non_foot_ground_contact(self):
        """Detect if any parts of the robot, other than the feet, are touching the ground. Returns number of non-foot
        contacts."""

        num_contact_points = 0
        for i in range(self.num_links):
            if i in self.foot_links: # the feet themselves are allow the touch the ground
                continue
            points = self.client.getContactPoints(bodyA=self.quadruped, linkIndexA=i)
            num_contact_points += len(points)
        return num_contact_points

    def load_urdf(self, fixed=False, fixed_position=[0, 0, 1.0],
                  fixed_orientation=[0, 0, 0]):
        urdfFlags = p.URDF_USE_SELF_COLLISION
        path = str(os.path.dirname(__file__)) + '/urdf/aliengo.urdf'
        quats = self.client.getQuaternionFromEuler(fixed_orientation)
        if fixed:
            quadruped = self.client.loadURDF(
                path,
                basePosition=fixed_position,
                baseOrientation=quats,
                flags=urdfFlags,
                useFixedBase=True)
        else:
            quadruped = self.client.loadURDF(path,
                                             basePosition=[0, 0, 0.48],
                                             baseOrientation=[0, 0, 0, 1],
                                             flags=urdfFlags,
                                             useFixedBase=False)

        self.foot_links = [5, 9, 13, 17]

        for i in range (self.client.getNumJoints(quadruped)):
            self.client.changeDynamics(quadruped, i, linearDamping=0, angularDamping=.5)

        # self.quadruped.foot_links = [5, 9, 13, 17]
        # self.lower_legs = [2,5,8,11]
        # for l0 in self.lower_legs:
        #     for l1 in self.lower_legs:
        #         if (l1>l0):
        #             enableCollision = 1
        #             # print("collision for pair",l0,l1, self.client.getJointInfo(self.quadruped,l0, physicsClientId=self.client)[12],p.getJointInfo(self.quadruped,l1, physicsClientId=self.client)[12], "enabled=",enableCollision)
        #             self.client.setCollisionFilterPair(self.quadruped, self.quadruped, l0,l1,enableCollision, physicsClientId=self.client)

        return quadruped

    def remove_body(self):
        self.client.removeBody(self.quadruped)

    def self_collision(self):
        '''Returns number of robot self-collision points.'''

        points = self.client.getContactPoints(self.quadruped, self.quadruped)
        return len(points)

    def set_joint_position_targets(self, positions, true_positions=False):
        """Take positions in range of [-1, 1] or true positions.
        Map these to true joint positions and set them
        as joint position targets.
        """
        assert isinstance(positions, np.ndarray)
        if not true_positions:
            assert ((-1.0 <= positions) & (positions <= 1.0)).all(), \
                '\nposition received: ' + str(positions) + '\n'
            positions = self.actions_to_positions(positions)

        self.client.setJointMotorControlArray(
            self.quadruped,
            self.motor_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.max_torque * np.ones(self.n_motors),
            positionGains=self.kp * np.ones(self.n_motors),
            velocityGains=self.kd * np.ones(self.n_motors))

        # self.true_joint_position_target_history.pop()
        # self.true_joint_position_target_history.insert(0, positions)


    def positions_to_actions(self, positions):
        """Maps actual robot joint positions in radians
        to the range of [-1.0, 1.0]
        """
        return (positions - self.position_mean) / (self.position_range * 0.5)

    def actions_to_positions(self, actions):
        """Take actions (normalized positions) in the range
        of [-1.0, 1.0] and maps them to actual joint positions in
        radians.
        """
        return actions * (self.position_range * 0.5) + self.position_mean

    def _find_position_bounds(self):
        positions_lb = np.zeros(self.n_motors)
        positions_ub = np.zeros(self.n_motors)
        # find bounds of action space
        for i in range(self.n_motors):
            joint_info = self.client.getJointInfo(self.quadruped, self.motor_joint_indices[i])
            # bounds on joint position
            positions_lb[i] = joint_info[8]
            positions_ub[i] = joint_info[9]

        # no joint limits given for the thigh joints, so set them to plus/minus 90 degrees
        for i in range(self.n_motors):
            if positions_ub[i] <= positions_lb[i]:
                positions_lb[i] = -3.14159 * 0.5
                positions_ub[i] = 3.14159 * 0.5

        position_mean = (positions_ub + positions_lb)/2
        position_range = positions_ub - positions_lb

        return positions_lb, positions_ub, position_mean, position_range

    # def get_joint_states(self):
    #     '''Note: Reaction forces will return all zeros unless a torque sensor has been set'''

    #     joint_states = self.client.getJointStates(self.quadruped, self.motor_joint_indices)
    #     joint_positions  = self._positions_to_actions(np.array([joint_states[i][0] for i in range(self.n_motors)]))
    #     joint_velocities = np.array([joint_states[i][1] for i in range(self.n_motors)])
    #     reaction_forces  = np.array([joint_states[i][2] for i in range(self.n_motors)])
    #     applied_torques  = np.array([joint_states[i][3] for i in range(self.n_motors)])
    #     return joint_positions, joint_velocities, reaction_forces, applied_torques


    # def get_base_position_and_orientation(self):
    #     base_position, base_orientation = self.client.getBasePositionAndOrientation(self.quadruped)
    #     return np.array(base_position), np.array(base_orientation)


    # def get_base_twist(self):
    #     return np.array(self.client.getBaseVelocity(self.quadruped)).flatten()

    def generate_reset_joint_positions(self, stochastic=True, true_positions=True):
        """Returns the quadruped positions corresponding to the  default starting position, knees slightly bent,
        from first line of mocap file"""

        positions = np.array([0.037199,    0.660252,   -1.200187,   -0.028954,    0.618814,
                            -1.183148,    0.048225,    0.690008,   -1.254787,   -0.050525,    0.661355,   -1.243304])
        if stochastic:
            positions += (np.random.random_sample(12) - 0.5) * self.position_range * 0.05
            positions = np.clip(positions, self.positions_lb, self.positions_ub)
        if not true_positions: # convert to true positions vs normalized positions
            positions = self.positions_to_actions(positions)
        return positions



    def reset_joint_positions(self, positions, true_positions=True):
        """
        This ignores any physics or controllers and just overwrites joint positions to the given value.
        Returns the foot positions in foot frame, for use as an initial observation in PMTG controllers.
        NOTE: I am assuming that this function is always called during env.reset(). I am using it to
        initialize self.foot_target_history and self.phases
        """

        if not true_positions:
            positions = self.actions_to_positions(positions)


        for i in range(self.n_motors): # for some reason there is no p.resetJointStates (plural)
            self.client.resetJointState(self.quadruped,
                                self.motor_joint_indices[i],
                                positions[i],
                                targetVelocity=0)

        self.client.setJointMotorControlArray(self.quadruped,
                                    self.motor_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=positions,
                                    forces=self.max_torque * np.ones(self.n_motors),
                                    positionGains=self.kp * np.ones(12),
                                    velocityGains=self.kd * np.ones(12))


        # TODO should the below code just go to reset state?
        self.foot_target_history = [self.get_foot_frame_foot_positions()] * 3
        self.phases = np.array([0, np.pi, np.pi, 0])
        self.f_i = np.zeros(4)
        self.joint_pos_error_history = [np.zeros(12)] * 3
        self.joint_velocity_history = [np.zeros(12)] * 3
        self.true_joint_position_target_history = [positions] * 3
        return self.foot_target_history[0]



    # def _vis_coordinate_system(self):
    #     self.line_length = 0.25
    #     for i in range(4):
    #         self.client.addUserDebugLine([0,0,0], [self.line_length,0,0], lineColorRGB=[1,0,0],
    #                                     parentLinkIndex=self.thigh_links[i], replaceItemUniqueId=self.line_ids[i][0],
    #                                     parentObjectUniqueId=self.quadruped,
    #                                     lineWidth=self.line_width)
    #         self.client.addUserDebugLine([0,0,0], [0,self.line_length,0], lineColorRGB=[0,1,0],
    #                                     parentLinkIndex=self.thigh_links[i], replaceItemUniqueId=self.line_ids[i][1],
    #                                     parentObjectUniqueId=self.quadruped,
    #                                     lineWidth=self.line_width)
    #         self.client.addUserDebugLine([0,0,0], [0,0,self.line_length], lineColorRGB=[0,0,1],
    #                                     parentLinkIndex=self.thigh_links[i], replaceItemUniqueId=self.line_ids[i][2],
    #                                     parentObjectUniqueId=self.quadruped,
    #                                     lineWidth=self.line_width)


def sine_tracking_test():
    # test foot position command tracking and print tracking error

    from env import AliengoEnv
    import yaml
    import time
    import torch
    np.random.seed(0)
    torch.manual_seed(0)

    path = os.path.join(os.path.dirname(__file__),
                        '../config/state_pmtg.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = True
    params['fixed_position'] = [0.0, 0.0, 1.5]
    env = AliengoEnv(**params)
    env.reset()
    t = 0
    counter = 0
    while True:
        command = np.array([[0.1 * np.sin(2*t),
                             -0.1 * np.sin(2*t),
                             -0.3 + 0.1 * np.sin(2*t)] for _ in range(4)])
        env.quadruped.set_foot_positions(command)
        orientation = env.client.getQuaternionFromEuler(
            [np.pi / 4.0 * np.sin(t)] * 3)
        env.client.resetBasePositionAndOrientation(env.quadruped.quadruped,
                                                   [-1, 1, 1],
                                                   orientation)

        # if counter % 240 == 0:
        #     env.quadruped.apply_foot_disturbance()
        env.client.stepSimulation()
        time.sleep(1/240.)
        t += 1/240.0 * 0.1
        counter += 1
        calculate_tracking_error(command, env.client, env.quadruped)



def floor_tracking_test():
    # test foot position command tracking and print tracking error
    from env import AliengoEnv
    import yaml
    import time
    import torch
    np.random.seed(0)
    torch.manual_seed(0)

    path = os.path.join(os.path.dirname(__file__),
                        '../config/state_pmtg.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = True
    params['fixed_position'] = [0.0, 0.0, 1.5]
    env = AliengoEnv(**params)
    env.reset()
    t = 0
    while True:
        z = -0.450  # decreasing this to -0.51 should show feet collision with ground and inability to track
        command = np.array([[0.1 * np.sin(2*t), 0, z] for _ in range(4)])
        env.quadruped.set_foot_positions(command)
        env.client.resetBasePositionAndOrientation(env.quadruped.quadruped,
                                                   [0., 0., 0.5],
                                                   [0., 0., 0., 1.0])
        env.client.stepSimulation()
        time.sleep(1/240.)
        t += 1/240.
        calculate_tracking_error(command, env.client, env.quadruped)


def calculate_tracking_error(commanded_foot_positions, client, quadruped):
    errors = abs(commanded_foot_positions
                 - quadruped.get_foot_frame_foot_positions())
    print('Maximum tracking error: {:e}'.format(errors.max()))
    # print('Mean tracking error: {:e}'.format(errors.mean()))
    # print(commanded_global_foot_positions - actual_pos)
    print()


def plot_trajectory():
    import matplotlib.pyplot as plt
    h = 0.2
    k1 = np.linspace(0, 1, 100)
    z1 = h * (-2*k1*k1*k1 + 3*k1*k1) - 0.5
    k2 = np.linspace(1, 2, 100)
    z2 = h * (2*k2*k2*k2 - 9*k2*k2 + 12*k2 - 4) - 0.5
    k3 = np.linspace(-2, 0, 100)
    z3 = np.ones(100) * -0.5
    plt.plot(k1, z1, k2, z2, k3, z3)
    plt.show()
    sys.exit()


# def trajectory_generator_test(client, quadruped):
#     t = 0
#     counter = 1
#     # quadruped.reset_joint_positions(stochastic=False)
#     # time.sleep(2)
#     while True:
#         quadruped.set_trajectory_parameters(t, f=1.00)
#         client.stepSimulation()
#         time.sleep(1/240. * 1)
#         # if counter% 2 == 0:
#         #     calculate_tracking_error(command, client, quadruped)
#         t += 1/240.
#         counter += 1


def axes_shift_function_test():
    # generate a bunch of random points, shift them, shift back, calculate error

    from env import AliengoEnv
    import yaml

    path = os.path.join(os.path.dirname(__file__),
                        '../config/TEST_pmtg_env.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params['fixed'] = True
    env = AliengoEnv(**params)
    quadruped = env.quadruped
    client = env.client
    n = 1000
    # distributed U[-10, 10)
    test_points = (np.random.random_sample((n, 4, 3)) - 0.5) * 20
    error = np.zeros(n)
    for i in range(n):  # loop bc these methods are not vectorized
        output = quadruped.foot_frame_pos_to_global(
            quadruped.get_foot_frame_foot_positions(
                global_pos=test_points[i]
            )
        )
        error[i] = abs(output - test_points[i]).max()
    print('\n' + '#' * 50)
    print('Max Error One way: {}'.format(error.max()))

    error = np.zeros(n)
    for i in range(n):
        output = quadruped.get_foot_frame_foot_positions(
            global_pos=quadruped.foot_frame_pos_to_global(
                test_points[i]
            )
        )
        error[i] = abs(output - test_points[i]).max()
    print('Max Error Other way: {}'.format(error.max()))
    print('#' * 50 + '\n')
    # Now test with actual robot measurements
    max_error = -1000
    for i in range(10):
        # breakpoint()
        pos = np.random.random_sample(3) * 5
        orn = client.getQuaternionFromEuler((
            np.random.random_sample(3) - 0.5) * np.pi)
        client.resetBasePositionAndOrientation(
            quadruped.quadruped,
            posObj=pos,
            ornObj=orn
        )
        client.stepSimulation()
        truth_global_pos = np.array([i[0] for i in client.getLinkStates(
            quadruped.quadruped, quadruped.foot_links)])
        truth_global_pos[:, 2] -= 0.0265
        temp = quadruped.get_foot_frame_foot_positions(
            global_pos=truth_global_pos)
        output = quadruped.foot_frame_pos_to_global(temp)
        error = abs(output - truth_global_pos).max()
        if error > max_error:
            max_error = error
    print("Max error with real robot measurements: {}".format(max_error))


def test_disturbances(client, quadruped):
    client.setRealTimeSimulation(0)
    client.removeBody(quadruped.quadruped)
    quadruped = Aliengo(client, fixed=False)

    # cstr = client.createConstraint(parentBodyUniqueId=quadruped.quadruped,
    #                         parentLinkIndex=-1,
    #                         childBodyUniqueId=-1,
    #                         childLinkIndex=-1,
    #                         jointType=p.JOINT_FIXED,
    #                         jointAxis=[0]*3,
    #                         parentFramePosition=[0]*3,
    #                         childFramePosition=[0,0,1])
    # client.changeConstraint(cstr, maxForce=100.0)
    counter = 0
    flag = True
    while True:
        if counter % 100 == 0:
            if flag:
                print(quadruped.apply_torso_disturbance())  # wrench=[1e10]*6))
                flag = False
            else:
                print(quadruped.apply_foot_disturbance())
                flag = True
        time.sleep(1./240)
        client.stepSimulation()
        counter += 1


def test_calf_joint_torques(client, quadruped):
    # quadruped.reset_joint_positions(stochastic=False)
    quadruped.foot_target_history = [quadruped.get_foot_frame_foot_positions()] * 3
    quadruped.phases = np.array([0, np.pi, np.pi, 0])
    quadruped.f_i = np.zeros(4)
    quadruped.joint_pos_error_history = [np.zeros(12)] * 3
    quadruped.joint_velocity_history = [np.zeros(12)] * 3
    quadruped.last_true_joint_position_targets = np.zeros(12) #TODO


    hip_offset_from_base = client.getJointInfo(quadruped.quadruped, quadruped.hip_joints[0])[14] #TODO just store this value
    base_p, base_o = client.getBasePositionAndOrientation(quadruped.quadruped)
    hip_joint_position, _ = np.array(client.multiplyTransforms(positionA=base_p,
                                            orientationA=base_o,
                                            positionB=hip_offset_from_base,
                                            orientationB=[0.0, 0.0, 0.0, 1.0]))
    pos = np.array(hip_joint_position) - np.array([0,0,1])
    # cstr = client.createConstraint(quadruped.quadruped, quadruped.thigh_links[0], -1, -1, p.JOINT_FIXED,
                                                                                                # [0,0,0], [0,0,0], pos)
    # client.changeConstraint(cstr, maxForce=1e10)
    while True:
        client.resetJointState(quadruped.quadruped, quadruped.thigh_joints[0], 0)
        client.resetJointState(quadruped.quadruped, quadruped.hip_joints[0], 0)
        force = [0,0, 0]
        torque = [10,0, 0]
        # client.applyExternalForce(quadruped.quadruped, quadruped.shin_links[0], force, [0,0,0], p.WORLD_FRAME)
        client.applyExternalTorque(quadruped.quadruped, quadruped.shin_links[0], torque, p.WORLD_FRAME)
        client.stepSimulation()


        quadruped.update_state(flat_ground=True)
        print(quadruped.reaction_forces[2], end='\n\n')
        time.sleep(1.0/240)


def test_trajectory_generator():
    from env import AliengoEnv
    import yaml
    import time
    import torch
    np.random.seed(0)
    torch.manual_seed(0)

    path = os.path.join(os.path.dirname(__file__),
                        '../config/state_pmtg.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = True
    params['fixed_position'] = [0.0, 0.0, 1.5]
    env = AliengoEnv(**params)
    env.reset()
    """The action should consist of amplitude, walking_height,
        frequency, plus 12 joint positions residuals,
        all in the range of [-1, 1]
        """
    amplitude = np.array([0.0])
    walking_height = np.array([-1.0])
    frequency = np.array([1.0])
    residuals = np.zeros(12)
    action = np.concatenate((amplitude, walking_height, frequency, residuals))
    # for _ in range(100):
    #     env.client.stepSimulation()
    time.sleep(1.0)
    while True:
        time.sleep(1/240. * env.action_repeat)
        env.step(action)
        if action[0] < 1.00:
            action[0] += 0.01


def check_foot_position_reach():
    from env import AliengoEnv
    import yaml
    import time

    path = os.path.join(os.path.dirname(__file__),
                        '../config/default_footstep.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = True
    params['fixed_position'] = [0.0, 0.0, 2.0]
    env = AliengoEnv(**params)
    env.reset()
    while True:
        time.sleep(1.0 / 240.0 * 10.0)
        env.client.stepSimulation()
        env.step(np.ones(12))
        pos = env.quadruped.get_foot_frame_foot_positions()[0]
        print('{:.3f}, {:.3f}, {:.3f}'.format(*pos))


if __name__ == '__main__':
    axes_shift_function_test()
    # check_foot_position_reach()
    # test_trajectory_generator()
    # sine_tracking_test()
    # floor_tracking_test()





    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # import sys; sys.exit()
    # # plot_trajectory()

    # # set up the quadruped in a pybullet simulation
    # from pybullet_utils import bullet_client as bc
    # client = bc.BulletClient(connection_mode=p.GUI)
    # client.setTimeStep(1/240.)
    # client.setGravity(0,0,-9.8)
    # client.setRealTimeSimulation(0) # this has no effect in DIRECT mode, only GUI mode
    # plane = client.loadURDF('./urdf/plane.urdf')
    # # set kp = 1.0 just for when I'm tracking, to eliminate it as a *large* source of error
    # quadruped = AliengoQuadruped(client, fixed=True, fixed_orientation=[0] * 3, fixed_position=[0.15,-0.15,0.7], kp=1.0,
    #                     vis=False, gait_type='trot')

    # trajectory_generator_test(client, quadruped) # tracking performance is easily increased by setting kp=1.0
    # axes_shift_function_test(client, quadruped) # error should be about 2e-17
    # test_disturbances(client, quadruped) # unfix the base to actually see results of disturbances
    # quadruped.reset_joint_positions()
    # while True:
    #     begin = time.time()
    #     time.sleep(1./240)
    #     quadruped._get_foot_terrain_scan(flat_ground=True)
    #     client.stepSimulation()
    #     print(time.time() - begin)


    # # test_calf_joint_torques(client, quadruped)

    # foot_positions = np.zeros((4, 3))
    # lateral_offset = 0.11
    # foot_positions[:,-1] = -0.4
    # foot_positions[[0,2], 1] = -lateral_offset
    # foot_positions[[1,3], 1] = lateral_offset
    # quadruped.reset_joint_positions()
    # quadruped._wide_step_rew()
    # while True:
    #     client.stepSimulation()
    #     quadruped.set_foot_positions(foot_positions)
    #     print(quadruped.self_collision(), 'asdf')
    #     time.sleep(1./240)

    # while True:
    #     quadruped.get_privledged_terrain_info(client)
    #     client.stepSimulation()
    #     time.sleep(1/240.)
    # quadruped.reset_joint_positions()
    # quadruped.update_state()
    # while True:
    #     time.sleep(1/240.)
    #     quadruped.update_state()
    #     print(np.array(list(client.getEulerFromQuaternion(quadruped.base_orientation)))*180.0/np.pi)
    #     client.stepSimulation()
    # quadruped.get_hutter_teacher_pmtg_observation(flat_ground=True)
