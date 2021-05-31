import argparse
import os

from stable_baselines3.common.running_mean_std import RunningMeanStd
import numpy as np
from gym.wrappers.clip_action import ClipAction
from aliengo_env.env import AliengoEnv

from train import get_params


parser = argparse.ArgumentParser(description='Script for visualizing trained models')
parser.add_argument('--config', default='defaults', type=str)
# parser.add_argument('--save-vid', default=False, action='store_true')
# parser.add_argument('--deterministic', action='store_true', default=False, help='whether to use a deterministic policy')
parser.add_argument('--vis', default=False, action='store_true', help='enable diagnostic visualization of aliengo')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--ws', type=int, default=-1)
parser.add_argument('--speed', default=1.0, type=float)
parser.add_argument('--device', default='cpu', help='device to load the trained model into')
args = parser.parse_args()

filename = os.path.join('./trained_models', args.config + str(args.seed) + '.npz')
if args.ws != -1:
    import paramiko
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ws_ip = ['143.215.128.18',
             '143.215.131.33',
             '143.215.184.72',
             '143.215.128.16',
             '143.215.131.25',
             '143.215.131.23']
    print('\n\nOpening Remote SSH Client...\n\n')
    ssh_client.connect(ws_ip[args.ws - 1], 22, 'jcoholich')
    print('Connected!\n\n')
    # ssh_client.exec_command('cd hutter_kostrikov; cd trained_models')
    sftp_client = ssh_client.open_sftp()
    filename = sftp_client.open(os.path.join('aliengo_baseline/', filename), 'rb')
temp = np.load(filename)
mean_std = RunningMeanStd()
policy = temp['arr_0']
mean_std.mean = temp['arr_1']
mean_std.var = temp['arr_2']
print('Agent Loaded\n\n')


yaml_args = get_params(args.config)

# yaml_args.render = True
yaml_args.env_params['render'] = True
yaml_args.env_params['vis'] = args.vis

if yaml_args.env_name == "aliengo":
    env = AliengoEnv(**yaml_args.env_params)
    env = ClipAction(env)
else:
    raise NotImplementedError

obs = env.reset()
while True:
    norm_obs = (obs - mean_std.mean) / np.sqrt(mean_std.var)
    action = policy @ np.expand_dims(norm_obs, 1)
    obs, rew, done, _ = env.step(action.flatten())
    if done:
        obs = env.reset()
