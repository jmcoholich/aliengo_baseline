
"""
I want the command to look like:
python enjoy.py --config default --save-video

# TODO trained models should be named the same as their yaml file
# TODO ensure that when I overwrite a saved model, I meant to
# TODO add a way to make train and test envs different (perhaps load a different env yaml file)
# TODO add a way to just change the random seed
# TODO add way to specify difficulty
"""


import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import time
import pybullet as p
import cv2

from train import get_params


'''
python enjoy.py --load-dir trained_models/ppo --env-name "gym_aliengo:Aliengo-v0" --seed 4581852999
python enjoy.py --load-dir trained_models/ppo --env-name "gym_aliengo:AliengoSteppingStones-v0" --seed 31354685
python enjoy.py --load-dir trained_models/ppo --env-name "gym_aliengo:AliengoHills-v0" --seed 5468319
python enjoy.py --load-dir trained_models/ppo --env-name "gym_aliengo:AliengoSteps-v0" --seed 94183919
python enjoy.py --load-dir trained_models/ppo --env-name "gym_aliengo:AliengoStairs-v0" --seed 465797972
'''


def add_frame(render_func, img_array):
    img = render_func('rgb_array')
    height, _, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.putText(np.float32(img), ('%f'% VIDEO_SPEED).rstrip('0') + 'x Speed'  , (1, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
    img = cv2.putText(np.float32(img), '%d FPS' % FPS , (1, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
    img = np.uint8(img)
    img_array.append(img)
    return img_array


def write_video(img_array, env_name, fps, args):
    height, width, layers = img_array[0].shape
    size = (width, height)
    if not os.path.exists('videos/' + env_name):
        os.makedirs('videos/' + env_name)

    filename = os.path.join('videos', env_name, str(args.seed) + '.avi' )
    print(filename)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for img in img_array:
        out.write(img)
    out.release()
    print('Video saved')

sys.path.append('a2c_ppo_acktr')  # TODO

# load yaml #TODO use function from train.py


parser = argparse.ArgumentParser(description='Script for visualizing trained models')
parser.add_argument('--config', default='defaults',
                     help='Specify yaml file corresponding to env and trained agent you wish to load.')
parser.add_argument('--save-vid', default=False, action='store_true')
parser.add_argument('--deterministic', action='store_true', default=False, help='whether to use a deterministic policy')
parser.add_argument('--vis', default=False, action='store_true', help='enable diagnostic visualization of aliengo')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--ws', type=int, default=-1)
parser.add_argument('--speed', default=1.0, type=float)
parser.add_argument('--device', default='cpu', help='device to load the trained model into')



'''
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
# parser.add_argument(
#     '--env-name',
#     default='gym_aliengo:Aliengo-v0',
#     help='environment to train on (default: gym_aliengo:Aliengo-v0)')
parser.add_argument(
    '--train-env',
    default='gym_aliengo:Aliengo-v0',
    help='environment to train on (default: gym_aliengo:Aliengo-v0)')
parser.add_argument(
    '--test-env',
    default=None,
    help='environment to train on (default: args.train_env)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--difficulty',
    default=0.25,
    type=float,
    help='difficulty of the test_env from 0 to 1, easy to hard. Default 0.25')
parser.add_argument(
    '--kp',
    default=1.0,
    type=float)
parser.add_argument(
    '--kd',
    default=1.0,
    type=float)
parser.add_argument(
    '--gait-type',
    default='trot')
parser.add_argument(
    '--env-mode',
    default='hutter_teacher_pmtg'
)
parser.add_argument( #TODO
    '--cherry-pick',
    default=0,
    type=int,
    help='number of runs to test before selecting the max return one. Automatically turns off deterministic'
)
'''


# parser.add_argument(
#     '--seed',
#     default=1,
#     help='seed used in training desired policy to visualize')
args = parser.parse_args()

# if args.test_env is None:
#     args.test_env = args.train_env

# args.det = not (args.non_det or args.cherry_pick)

# env_kwargs = {'env_mode':args.env_mode,
#                     'set_difficulty': args.difficulty,
#                     'vis':args.vis,
#                     'kp': args.kp,
#                     'kd': args.kd,
#                     'gait_type': args.gait_type}

yaml_args = get_params(args.config)
yaml_args.env_params['render'] = not args.save_vid
yaml_args.env_params['vis'] = args.vis

env = make_vec_envs(
    yaml_args.env_name,
    args.seed,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False,
    # render=not args.save_vid,
    env_params=yaml_args.env_params)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
filename = os.path.join('./trained_models', args.config + str(args.seed) + '.pt')
if args.ws == -1:
    actor_critic, obs_rms, _, _ = torch.load(filename, args.device)
else:
    import paramiko
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ws_ip = ['143.215.184.71',
             '143.215.184.13',
             '143.215.184.72',
             '143.215.128.16',
             '143.215.131.25',
             '143.215.131.23']
    print('\n\nOpening Remote SSH Client...\n\n')
    ssh_client.connect(ws_ip[args.ws - 1], 22, 'jcoholich')
    print('Connected!\n\n')
    # ssh_client.exec_command('cd hutter_kostrikov; cd trained_models')
    sftp_client = ssh_client.open_sftp()
    remote_file = sftp_client.open(os.path.join('aliengo_baseline/', filename), 'rb')
    actor_critic, obs_rms, _, _ = torch.load(remote_file, map_location=args.device)
    print('Agent Loaded\n\n')


    # # copy the file to local and then run instead
    # sftp_client.get(os.path.join('hutter_kostrikov/trained_models',filename), os.path.join('/home/jeremiah/hutter_kostrikov/trained_models',filename))
    # actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, filename))
# actor_critic.to('cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
# TODO figure this shit out below
# if render_func is not None:
#     render_func('human')

#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "torso"):
#             torsoId = i

if args.save_vid:
    assert 240.0%env.venv.venv.envs[0].action_repeat == 0
    global FPS
    global VIDEO_SPEED
    FPS = int(240.0/env.venv.venv.envs[0].action_repeat)
    VIDEO_SPEED = 1.0
    img_array = []
    counter = 0
    img_array = add_frame(render_func, img_array)
    total_rew = 0.0

if yaml_args.env_name == 'aliengo':
    loop_time = 1/240. * env.venv.venv.envs[0].action_repeat * 1.0/args.speed
else:
    loop_time = 0.01

episode_rew = 0.0
while True:
    time.sleep(loop_time)
    with torch.no_grad():
        value, action, _, recurrent_hidden_states, _ = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.deterministic)

    # Obser reward and next obs
    if 'Box' in str(env.action_space):
        obs, reward, done, info = env.step(action)

    else:
        obs, reward, done, info = env.step(action)
    print(obs)
    print()
    episode_rew += reward
    if done:
        print('Episode reward: {}'.format(episode_rew.item()))
        if 'termination_reason' in info[0].keys():
            print(info[0]['termination_reason'])
        else:
            print('Timeout')
        episode_rew = 0.0
    masks.fill_(0.0 if done else 1.0)

    if yaml_args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')

    if args.save_vid:
        img_array = add_frame(render_func, img_array)
        counter += 1
        total_rew += reward

    if args.save_vid and done:
        break

if args.save_vid:
    write_video(img_array, yaml_args.env_name, FPS, args)
