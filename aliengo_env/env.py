import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pybullet as p
import os
import time
import numpy as np
import warnings
from cv2 import putText, FONT_HERSHEY_SIMPLEX
from . import aliengo_quadruped
from pybullet_utils import bullet_client as bc


from .modular_obs_and_act import Observation, Action
from .obstacles.hills import Hills
from .obstacles.steps import Steps  
from .obstacles.stepping_stones import SteppingStones
from .obstacles.stairs import Stairs
from .reward import RewardFunction

class AliengoEnv(gym.Env):
    def __init__(self, 
                    render=False, 
                    # env_mode='pmtg',
                    apply_perturb=False,
                    avg_time_per_perturb=5.0, # seconds
                    action_repeat=4,
                    timeout=60.0, # number of seconds to timeout after
                    vis=False,
                    observation_parts=['joint_torques', 'joint_positions', 'joint_velocities', 'IMU'],
                    action_parts=['joint_positions'],
                    reward_parts=['forward_velocity'],
                    obstacles=None,
                    **quadruped_kwargs):
                    # fixed=False,
                    # fixed_position=[0,0,1.0], 
                    # fixed_orientation=[0,0,0],
                    # gait_type='trot'):
        self.apply_perturb = apply_perturb
        self.avg_time_per_perturb = avg_time_per_perturb # average time in seconds between perturbations 
        self.n_hold_frames = action_repeat
        self.eps_timeout = 240.0/self.n_hold_frames * timeout # number of steps to timeout after
        # if U[0, 1) is less than this, perturb
        self.perturb_p = 1.0/(self.avg_time_per_perturb * 240.0) * self.n_hold_frames
        self.quadruped_kwargs = quadruped_kwargs
        print('self.quadruped_kwargs in aliengo_env', self.quadruped_kwargs)
        # setting these to class variables so that child classes can use it as such. 
        self.vis = vis

        if render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) 
        if self.client == -1 or self.fake_client == -1: 
            # not sure that BulletClient would return -1 for failure, but leaving this for now
            raise RuntimeError('Pybullet could not connect to physics client')

        self.plane = self.client.loadURDF(str(os.path.dirname(__file__)) +  '/urdf/plane.urdf')
        self.quadruped = aliengo_quadruped.AliengoQuadruped(pybullet_client=self.client, 
                                            vis=self.vis,
                                            **self.quadruped_kwargs)
        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(False) # setting this to True messes with things
        self.client.setTimeStep(1/240.0)
        
        self.observe = Observation(observation_parts, self.quadruped)
        self.observation_space = spaces.Box(
            low=self.observe.observation_lb,
            high=self.observe.observation_ub,
            dtype=np.float32)
        
        self.act = Action(action_parts, self.quadruped)
        self.action_lb = self.act.action_lb
        self.action_ub = self.act.action_ub
        self.eps_step_counter = 0 # Used for triggering timeout
        self.mean_rew_dict = {} # used for logging the mean reward terms at the end of each episode
        self.action_space = spaces.Box(
            low=self.act.action_lb,
            high=self.act.action_ub,
            dtype=np.float32)


        obstacles_dict = {'hills': Hills, 'steps': Steps, 'stairs': Stairs, 'stepping_stones': SteppingStones}
        if obstacles is not None:
            self.obstacles = obstacles_dict[obstacles](self.client, self.fake_client)

        self.reward_func = RewardFunction(self.client, reward_parts, self.quadruped)
            

    def generate_disturbances(self):

        if (np.random.rand() < self.perturb_p) and self.apply_perturb: 
            '''TODO eventually make disturbance generating function that applies disturbances for multiple timesteps'''
            if np.random.rand() > 0.5:
                # TODO returned values will be part of privledged information for teacher training
                force, foot = self.quadruped.apply_foot_disturbance() 
            else:
                # TODO returned values will be part of privledged information for teacher training
                wrench = self.quadruped.apply_torso_disturbance()



    def step(self, action):
        DELTA = 0.0001 # this should just be for floating point errors
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 


        self.generate_disturbances()


        for _ in range(self.n_hold_frames):
            self.act(action)
            self.t += 1./240.
            self.client.stepSimulation()
            if self.vis: self.quadruped.visualize()

        self.eps_step_counter += 1
        self.quadruped.update_state(flat_ground=False, fake_client=self.fake_client)

        obs = self.observe()

        info = {}
        done, termination_dict = self.is_state_terminal() # this must come after self._update_state()
        info.update(termination_dict) # termination_dict is an empty dict if not done

        rew, rew_dict = self.reward_func()
        self.update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled']   = self.quadruped.base_position[0]
            info.update(self.mean_rew_dict)

        return obs, rew, done, info


    def update_mean_rew_dict(self, rew_dict):
        '''Update self.mean_rew_dict, which keeps a running average of all terms of the reward. At the end of the 
        episode, the average will be logged. '''

        if self.eps_step_counter == 1:
            for key in rew_dict:  
                self.mean_rew_dict['mean_' + key] = rew_dict[key]
        elif self.eps_step_counter > 1:
            for key in rew_dict:
                self.mean_rew_dict['mean_' + key] += \
                                        (rew_dict[key] - self.mean_rew_dict['mean_' + key])/float(self.eps_step_counter)
        else:
            assert False
            

    def reset(self, base_height=0.48, stochastic=True): #TODO make it so that I apply the torque at every simulation step
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. 
        
        Create new random terrain.
        '''

        self.eps_step_counter = 0
        self.t = 0.0
        ground_height = self.obstacles.reset()
        posObj = [0,0,base_height + ground_height] if ground_height is not None else [0,0,base_height]
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=posObj, 
                                            ornObj=[0,0,0,1.0]) 
        reset_position = self.quadruped.generate_reset_joint_positions(stochastic=stochastic)
        for i in range(50): # to let the robot settle on the ground.
            self.quadruped.reset_joint_positions(reset_position) 
            self.client.stepSimulation()
        self.quadruped.update_state(flat_ground=True, fake_client=self.fake_client) #TODO get rid of flatground
        obs = self.observe()

        return obs


    def render(self, mode='human', client=None):
        if client is None:
            client = self.client
        return self.quadruped.render(mode=mode, client=client)


    def is_state_terminal(self, flipping_bounds=[np.pi/2., np.pi/4., np.pi/4.], height_lb=0.23, height_ub=0.8):
        quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=flipping_bounds,
                                                                            height_lb=height_lb,
                                                                            height_ub=height_ub)
        timeout = (self.eps_step_counter >= self.eps_timeout) 
        if timeout:
            termination_dict['TimeLimit.truncated'] = True
        done = quadruped_done or timeout
        return done, termination_dict
            

if __name__ == '__main__':
    '''Perform check by feeding in the mocap trajectory provided by Unitree (linked) into the aliengo robot and
    save video. https://github.com/unitreerobotics/aliengo_pybullet'''

    import cv2
    env = gym.make('gym_aliengo:Aliengo-v0', use_pmtg=False)    
    env.reset()

    img = env.render('rgb_array')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_list = [img]
    counter = 0

    with open('mocap.txt','r') as f:
        for line_num, line in enumerate(f): 
            if line_num%2*env.n_hold_frames == 0: # Unitree runs this demo at 500 Hz. We run at 240 Hz, so double is close enough.
                action = env.quadruped._positions_to_actions(np.array(line.split(',')[2:],dtype=np.float32))
                obs,_ , done, _ = env.step(action)
                if counter%4 == 0:  # simulation runs at 240 Hz, so if we render every 4th frame, we get 60 fps video
                    img = env.render('rgb_array')
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_list.append(img)
                counter +=1 # only count the lines that are sent to the simulation (i.e. only count 
                # p.client.stepSimulation() calls)

    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter('test_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, size)
    for img in img_list:
        out.write(img)
    out.release()
    print('Video saved')


    
    



    


