# Aliengo Baseline

This is a codebase for trianing locomotion policies for the [Unitree Aliengo](https://www.unitree.com/products/aliengo/)
robot with [PyBullet](https://pybullet.org/wordpress/). As I have found, it is very difficult
to replicate reinforcement learning results for legged locomotion from scratch. Most
of the polices end up looking something like this:

<img src="videos/aliengo/output.gif" alt="drawing" width="400"/>

I later abandoned this pipeline in favor of using [NVIDIA's IsaacGym](https://developer.nvidia.com/isaac-gym).


This repo contains [this](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) PPO implementation and my own implementation of [Augmented Random Search](https://arxiv.org/abs/1803.07055).


## Installation

`git clone https://github.com/jmcoholich/aliengo_baseline.git`

`cd aliengo_baseline`

Install anaconda if you haven't already: https://www.anaconda.com/products/distribution

`conda create -n aliengo_baseline python=3.8.8`

`conda activate aliengo_baseline`

```pip install -r requirements.txt```

Install pytorch with

```pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html```


A policy can be trained with:

`python train.py`

Tested on Ubuntu 20.04
