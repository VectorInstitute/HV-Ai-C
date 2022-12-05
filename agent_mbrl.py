


from hnp.agent_hnp import HNPAgent, ObservationWrapper
from beobench.experiment.provider import create_env, config
import numpy as np
import os
import time
import gym
import itertools
import datetime
import pickle
import os
import logging

from numpy.lib import recfunctions
import pandas as pd

from beobench.experiment.provider import create_env, config
import numpy as np
import os
import time
import gym

if __name__ == "__main__":
    # Create environment and wrap observations
    obs_to_keep = np.array([6,7,8]) 
    lows = np.array([0,0,0])
    highs = np.array([5,1,1]) # if discrete then is number of actions
    mask = np.array([2,0,1]) 
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

