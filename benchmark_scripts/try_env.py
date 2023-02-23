import gym
import numpy as np
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import envs

import sinergym
from sinergym.utils.wrappers import LoggerWrapper

env = gym.make('Eplus-5Zone-hot-discrete-train-v1')
env = LoggerWrapper(env)

for i in range(1):
    obs = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    while not terminated:
        a = env.action_space.sample()
        obs, reward, terminated, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()
