import argparse
from warnings import warn as warn_

import torch
import yaml

from pytorch_a2c_ppo_acktr_gail.main import main as ppo_main

def warn(text):
    warn_('\033[93m' + text + '\033[0m')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        help="specify name of yaml config file. If none is given, use default.yaml",
                        type=str,
                        default="default")
    args = parser.parse_args()

    with open(r'config/defaults.yaml') as f:
        default_params = yaml.full_load(f)

    vars(args).update(default_params)
    if args.cuda and not torch.cuda.is_available():
        warn('No GPU found, running on CPU.')
        args.cuda = False

    return args



def main():
    args = get_params()
    ppo_main(args)


if __name__ == '__main__':
    main()

