import os
import argparse
from warnings import warn as warn_

import torch
import yaml

from pytorch_a2c_ppo_acktr_gail.main import main as ppo_main


def warn(text):
    warn_('\033[93m' + text + '\033[0m')


def get_params(config_yaml_path):
    with open(os.path.join('config', 'defaults.yaml')) as f:
        default_params = yaml.full_load(f)

    with open(os.path.join('config', config_yaml_path + '.yaml')) as f:
        params = yaml.full_load(f)

    default_params.update(params)  # this is a dict
    args = argparse.Namespace()
    vars(args).update(default_params)
    if args.cuda and not torch.cuda.is_available():
        warn('No GPU found, running on CPU.')
        args.cuda = False

    return args


def check_if_overwriting(config):
    """Ask user if they are sure they want to overwrite trained_model."""
    path = os.path.join('./trained_models', config + '.pt')
    if os.path.exists(path):
        warn("\nThe trained model for this config already exists "
             "on this machine. Overwrite it? (y/n)")
        answer = input().lower()
        if answer == 'y':
            return
        if answer == 'n':
            os.sys.exit()
            return


def main():  # TODO add a vis flag for training. (just to make sure the env is created correctly and whatnot)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="specify name of yaml config file. If none is given, use default.yaml",
                        type=str,
                        default="defaults")
    parser.add_argument("--resume",
                        help="loads a trained_model corresponding to the yaml file and resumes training.",
                        action="store_true",
                        default=False)
    parser.add_argument("--gpu-idx",
                        help="Specifies GPU to train on (if CUDA is enabled)",
                        type=int,
                        default=0)
    parser.add_argument("--seed",
                        type=int,
                        default=1)
    args = parser.parse_args()
    if args.resume:
        warn("Resuming training. Run can no longer be determininistically reproduced.")
    else:
        check_if_overwriting(args.config)
    main_args = get_params(args.config)
    ppo_main(main_args,
             args.config,
             args.seed,
             args.gpu_idx,
             resume=args.resume)


if __name__ == '__main__':
    main()
