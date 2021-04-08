import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate


class WandbLogWrapper:
    """
    This class is a wrapper for the wandb object. It averages batches of log_interval size until calling wandb.log(). 
    This is to save disk space in the wandb dir and speed up the loading of wandb pages. 
    NOTE: This does not handle the logging of strings like termination reason, etc.
    """
    def __init__(self, wandb, log_interval=100):
        self.wandb = wandb
        self.counter = 0
        self.mean_info = {}
        self.log_interval = log_interval


    def log(self, info):
        self.counter += 1
        if not self.mean_info: # if dict is empty, init
            self.mean_info = info
        else: # otherwise, update values (online mean algorithm)
            for key, value in info.items():
                if not (isinstance(value, str) or isinstance(value, bool)):
                    self.mean_info[key] += (value - self.mean_info[key])/float(self.counter)

        if self.counter == self.log_interval:
            self.wandb.log(self.mean_info)
            self.counter = 0


def main(args):
    total_env_steps = 0
    wandb.init(project=args.wandb_project, config=args)
    wandb_wrapper = WandbLogWrapper(wandb, log_interval=args.wandb_log_interval)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    if args.num_torch_threads:
        torch.set_num_threads(args.num_torch_threads) # TODO is there any reason to set this to one? 
    device = torch.device("cuda:" + str(args.gpu_idx) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, # TODO change the way I make envs
                         args.gamma, args.log_dir, device, False, env_params=args.env_params)

    action_ub = torch.from_numpy(envs.action_space.high).to(device)
    action_lb = torch.from_numpy(envs.action_space.low).to(device)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    mean_entropies = 0.0
    entropies_counter = 0.0

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]) #TODO return dist_entropy

            entropies_counter += 1.0
            mean_entropies += (dist_entropy.item() - mean_entropies)/entropies_counter


            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    wandb_info = {}
                    total_env_steps += info['episode']['l'] 
                    wandb_info['hours_wall_time'] = info['episode']['t']/3600.
                    wandb_info['episode_length']  = info['episode']['l'] 
                    wandb_info['episode_reward']  = info['episode']['r']
                    wandb_info['num_env_samples'] = total_env_steps 
                    wandb_info['average_entropy']  = mean_entropies
                    # these keys are already taken case of elsewhere 
                    # TODO eventually log a moving average of percentage/frequency of different termination conditions
                    blacklisted_keys = {'episode', 'bad_transition', 'termination_reason'} # don't log these
                    for key_of_logged_item, value_of_logged_item in info.items():
                        if key_of_logged_item not in blacklisted_keys:
                            wandb_info[key_of_logged_item] = value_of_logged_item
                    entropies_counter = 0.0 
                    wandb_wrapper.log(wandb_info)



            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
