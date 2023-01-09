import yaml
import sys

import numpy as np

from hnp.agents import QLearningAgent
from hnp.environment import ObservationWrapper, create_env


def main(config_path):
    # Create environment and wrap observations
    with open(config_path, "r") as conf_yml:
        config = yaml.safe_load(conf_yml)

    obs_to_keep = np.array([0, 1, 8]) 
    lows = np.array([0, 0, 0])
    highs = np.array([1, 1, 1])
    mask = np.array([0, 0, 0])

    env = create_env(config["env"])
    env = ObservationWrapper(env, obs_to_keep, lows, highs, mask)

    agent = QLearningAgent(
        env, 
        config["agent"]["params"],
        mask,
        lows,
        highs,
    )
    agent.train()
    agent.save_results()
    env.close()

if __name__ == "__main__":
    main(sys.argv[1])