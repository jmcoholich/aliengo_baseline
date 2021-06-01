import os
import argparse
from warnings import warn as warn_

import torch
import yaml

from pytorch_a2c_ppo_acktr_gail.main import main as ppo_main
from ars.ars_main import ars as ars_main


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


def check_if_overwriting(config, seed):
    """Ask user if they are sure they want to overwrite trained_model."""
    path = os.path.join('./trained_models', config + str(seed) + '.pt')
    if os.path.exists(path):
        warn("\nThe trained model for this config and seed already exists "
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
    # Added for sweep #########################################################
    # TODO
    # parser.add_argument("--lr", type=float, default=0.0002)
    # parser.add_argument("--n-dirs", type=int, default=1)
    # parser.add_argument("--delta-std", type=float, default=0.02)
    # parser.add_argument("--top-dirs-frac", type=float, default=1.00)

    ###########################################################################

    args = parser.parse_args()

    # args.top_dirs = int(args.top_dirs_frac * args.n_dirs) + 1  # ADDED for sweep

    if args.resume:
        warn("Resuming training. Run can no longer be determininistically reproduced.")
    else:
        check_if_overwriting(args.config, args.seed)
    main_args = get_params(args.config)

    # Added for sweep #########################################################
    # main_args.lr = args.lr
    # main_args.n_dirs = args.n_dirs
    # main_args.delta_std = args.delta_std
    # main_args.top_dirs = args.top_dirs
    ###########################################################################

    if main_args.algo == 'ppo':
        ppo_main(main_args,
                 args.config,
                 args.seed,
                 args.gpu_idx,
                 resume=args.resume)
    elif main_args.algo == 'ars':
        ars_main(main_args,
                 args.config,
                 args.seed)
    else:
        raise ValueError("Algo name invalid.")


if __name__ == '__main__':
    main()
