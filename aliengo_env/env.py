import os

from gym import spaces
import pybullet as p
import numpy as np
import cv2
import gym
from pybullet_utils import bullet_client as bc

import aliengo_quadruped
from action import Action
from observation import Observation
from termination import Termination
from obstacles.hills import Hills
from obstacles.steps import Steps
from obstacles.stepping_stones import SteppingStones
from obstacles.stairs import Stairs
from reward import RewardFunction
from utils import DummyObstacle


class AliengoEnv(gym.Env):
    def __init__(
            self,
            action_space: list,
            termination_conditions: list,
            observation_parts: list,
            reward_parts: list,
            stochastic_resets: bool,
            render=False,
            apply_perturb=False,
            avg_time_per_perturb=5.0,  # in seconds
            action_repeat=4,
            vis=False,
            obstacles=None,
            **quadruped_kwargs
    ):
        self.apply_perturb = apply_perturb
        self.stochastic_resets = stochastic_resets
        self.avg_time_per_perturb = avg_time_per_perturb
        self.action_repeat = action_repeat
        self.perturb_p = (1.0 / (self.avg_time_per_perturb * 240.0)
                          * self.action_repeat)
        self.quadruped_kwargs = quadruped_kwargs
        self.vis = vis

        if render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT)

        path = os.path.join(os.path.dirname(__file__), '/urdf/plane.urdf')
        self.plane = self.client.loadURDF(path)
        self.quadruped = aliengo_quadruped.AliengoQuadruped(
            pybullet_client=self.client,
            vis=self.vis,
            **self.quadruped_kwargs
        )
        self.client.setGravity(0, 0, -9.8)
        # setting this to True messes with things
        self.client.setRealTimeSimulation(False)
        self.client.setTimeStep(1 / 240.0)

        self.observe = Observation(observation_parts, self.quadruped)
        self.observation_space = spaces.Box(
            low=self.observe.observation_lb,
            high=self.observe.observation_ub,
            dtype=np.float32)

        self.act = Action(action_space, self.quadruped)
        self.action_lb = self.act.action_lb
        self.action_ub = self.act.action_ub
        self.eps_step_counter = 0  # Used for triggering timeout
        # used for logging the mean reward terms at the end of each episode
        self.mean_rew_dict = {}
        self.action_space = spaces.Box(
            low=self.act.action_lb,
            high=self.act.action_ub,
            dtype=np.float32)

        obstacles_dict = {'hills': Hills,
                          'steps': Steps,
                          'stairs': Stairs,
                          'stepping_stones': SteppingStones}
        if obstacles is not None:
            self.obstacles = obstacles_dict[obstacles](self.client,
                                                       self.fake_client)
        else:
            self.obstacles = DummyObstacle()
        self.reward_func = RewardFunction(self.client,
                                          reward_parts,
                                          self.quadruped)
        self.termination_func = Termination(termination_conditions,
                                            self.obstacles,
                                            self.quadruped,
                                            self)
        self.t = 0.0

    def generate_disturbances(self):
        if (np.random.rand() < self.perturb_p) and self.apply_perturb:
            # TODO eventually make disturbance generating function that
            #  applies disturbances for multiple timesteps
            if np.random.rand() > 0.5:
                force, foot = self.quadruped.apply_foot_disturbance()
                return (force, foot)
            else:
                wrench = self.quadruped.apply_torso_disturbance()
                return wrench

    def step(self, action):
        delta = 1e-4  # this should just be for floating point errors
        if not ((self.action_lb - delta <= action)
                & (action <= self.action_ub + delta)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n'
                             + str(self.action_lb) + '\nto\n'
                             + str(self.action_ub))

        self.generate_disturbances()

        for _ in range(self.action_repeat):
            self.act(action, self.t)
            self.t += 1.0 / 240.0
            self.client.stepSimulation()
            if self.vis:
                self.quadruped.visualize()

        self.eps_step_counter += 1
        self.quadruped.update_state(flat_ground=False,  # TODO flat_ground
                                    fake_client=self.fake_client)

        obs = self.observe()

        info = {}
        # this must come after quadruped._update_state()
        done, termination_dict = self.termination_func()
        # termination_dict is an empty dict if not done
        info.update(termination_dict)

        rew, rew_dict = self.reward_func()
        self.update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled'] = self.quadruped.base_position[0]
            info.update(self.mean_rew_dict)

        return obs, rew, done, info

    def update_mean_rew_dict(self, rew_dict):
        """Update self.mean_rew_dict, which keeps a running average of all
         terms of the reward. At the end of the
        episode, the dict will be returned as info by step().
        """

        if self.eps_step_counter == 1:
            for key in rew_dict:
                self.mean_rew_dict['mean_' + key] = rew_dict[key]
        elif self.eps_step_counter > 1:
            for key in rew_dict:
                temp = ((rew_dict[key] - self.mean_rew_dict['mean_' + key])
                        / float(self.eps_step_counter))
                self.mean_rew_dict['mean_' + key] += temp
        else:
            assert False

    def reset(self):
        """Reset the robot to a neutral standing position,
        knees slightly bent and generate new random terrain
        (if applicable).
        """

        if hasattr(self.quadruped, 'footstep_generator'):
            self.quadruped.footstep_generator.reset()
        self.eps_step_counter = 0
        self.t = 0.0
        self.obstacles.reset()
        base_height = 0.48
        pos = [0, 0, base_height + self.obstacles.ground_height]
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                                    posObj=pos,
                                                    ornObj=[0, 0, 0, 1.0])
        reset_position = self.quadruped.generate_reset_joint_positions(
            stochastic=self.stochastic_resets)
        for _ in range(50):  # to let the robot settle on the ground.
            self.quadruped.reset_joint_positions(reset_position)
            self.client.stepSimulation()
        self.quadruped.update_state(flat_ground=True,  # TODO get rid of flatground
                                    fake_client=self.fake_client)
        return self.observe()

    def render(self, mode='human', client=None):
        if client is None:
            client = self.client
        return self.quadruped.render(mode=mode, client=client)


def main(save_video):
    """Perform check by feeding in the mocap trajectory provided by Unitree
    (linked) into the aliengo robot and
    save video. https://github.com/unitreerobotics/aliengo_pybullet
    """

    import yaml

    path = os.path.join(os.path.dirname(__file__), '../config/TEST_env.yaml')
    with open(path) as f:
        params = yaml.full_load(f)

    if not save_video:
        params['render'] = True
    env = AliengoEnv(**params)
    env.reset()
    if save_video:
        img = env.render('rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_list = [img]
        counter = 0

    with open('mocap.txt', 'r') as f:
        for line_num, line in enumerate(f):
            # Unitree runs this demo at 500 Hz. We run at 240 Hz,
            # so double is close enough.
            if line_num % 2 * env.action_repeat == 0:
                action = env.quadruped.positions_to_actions(
                    np.array(line.split(',')[2:], dtype=np.float32))
                env.step(action)

                if save_video:
                    # simulation runs at 240 Hz, so rendering every 4th frame
                    # gives 60 fps video
                    if counter % 1 == 0:
                        img = env.render('rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img_list.append(img)
                    counter += 1
                    # only count the lines that are sent to the simulation
                    # (i.e. only count p.client.stepSimulation() calls)
    if save_video:
        height, width, _ = img.shape
        size = (width, height)
        out = cv2.VideoWriter(
            'test_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, size)
        for img in img_list:
            out.write(img)
        out.release()
        print('Video saved')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video", default=False)
    args = parser.parse_args()
    main(args.save_video)
