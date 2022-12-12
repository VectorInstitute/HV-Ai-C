from beobench.experiment.provider import create_env, config

from hnp.agent_hnp import HNPAgent, ObservationWrapper
import numpy as np
import os
import time
import gym

# Create environment and wrap observations
obs_to_keep = np.array([0, 8]) 
# lows = np.array([0,0,0,0,0,0])
# highs = np.array([1,1,1,8,1,9]) # if discrete then is number of actions
# mask = np.array([0,1,0,2,1,2]) # 0 = slow moving continuous, 1 = fast moving continuous, 2 = discrete
lows = np.array([0,0])
highs = np.array([1,1]) # if discrete then is number of actions
mask = np.array([0, 0]) 
env = create_env()

env = ObservationWrapper(env, obs_to_keep, lows, highs, mask)
agent = HNPAgent(
    env, 
    mask,
    lows,
    highs,
    eps_annealing=config["hyperparams"]["eps_annealing"], 
    learning_rate_annealing=config["hyperparams"]["lr_annealing"])

agent.learn(config["agent"]["config"]["num_episodes"], config["hyperparams"]["horizon"])
agent.save_results()
env.close()