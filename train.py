import argparse
from warnings import warn as warn_

import torch
import yaml
import os

from pytorch_a2c_ppo_acktr_gail.main import main as ppo_main

def warn(text):
    warn_('\033[93m' + text + '\033[0m')


def get_params(config_yaml_path):
    

    with open(os.path.join('config','defaults.yaml')) as f:
        default_params = yaml.full_load(f)

    with open(os.path.join('config',config_yaml_path + '.yaml')) as f:
        params = yaml.full_load(f)
    
    default_params.update(params) # this is a dict
    args = argparse.Namespace()
    vars(args).update(default_params)
    if args.cuda and not torch.cuda.is_available():
        warn('No GPU found, running on CPU.')
        args.cuda = False

    return args



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        help="specify name of yaml config file. If none is given, use default.yaml",
                        type=str,
                        default="defaults")
    args = parser.parse_args()
    main_args = get_params(args.config)
    ppo_main(main_args, args.config)


if __name__ == '__main__':
    main()

