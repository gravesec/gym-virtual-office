#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import gym_virtual_office
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)
    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)
    obs = env.reset()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)
    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)
    # print(obs)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
    elif event.key == 'backspace':
        reset()
    elif event.key == 'left':
        step(env.actions.west)
    elif event.key == 'right':
        step(env.actions.east)
    elif event.key == 'up':
        step(env.actions.north)
    elif event.key == 'down':
        step(env.actions.south)


parser = argparse.ArgumentParser()
parser.add_argument('--env', help='gym environment to load', default='VirtualOffice-v0')
parser.add_argument('--seed', type=int, help='random seed to generate the environment with', default=14)
parser.add_argument('--tile_size', type=int, help='size at which to render tiles', default=32)
parser.add_argument('--agent_view', default=False, help='draw the agent sees (partially observable view)', action='store_true')
args = parser.parse_args()

env = gym.make(args.env)
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)
reset()
# Blocking event loop
window.show(block=True)
