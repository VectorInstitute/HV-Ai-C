import yaml
import sys

import numpy as np

from hnp.agents import QLearningAgent
from hnp.environment import ObservationWrapper, create_env


def main(config_path):
    with open(config_path, "r") as conf_yml:
        config = yaml.safe_load(conf_yml)

    obs_to_keep = np.array(config["env"]["obs_to_keep"])
    mask = np.array(config["env"]["mask"])

    env = create_env(config["env"])
    env = ObservationWrapper(env, obs_to_keep)

    agent = QLearningAgent(
        env, 
        config["agent"],
        mask,
    )
    agent.train()
    agent.save_results()
    env.close()

if __name__ == "__main__":
    main(sys.argv[1])