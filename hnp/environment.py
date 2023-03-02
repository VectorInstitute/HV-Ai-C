"""
This file includes all environment related class and methods
"""

import gym
import numpy as np

from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
    RANGES_OFFICEGRID,
    RANGES_SHOP
)


class ObservationWrapper(gym.ObservationWrapper):
    """
    Sinergym environment wrapper to modify observations
    """

    def __init__(self, env, obs_to_keep):
        """
        Constructor for observation wrapper

        :param env: Sinergym environment
        :param obs_to_keep: Indices of state variables that are used

        :return: None
        """
        super().__init__(env)
        self.env = env
        if not obs_to_keep.any():
            self.obs_to_keep = np.arange(4, env.observation_space.shape[0])
        else:
            self.obs_to_keep = np.add(np.array(obs_to_keep), 4)

    def observation(self, observation):
        """
        Remove the unused state variables from observation

        :param observation: Full observation

        :return: Filtered observation
        """
        return observation[self.obs_to_keep]


def create_env(env_config: dict = None) -> gym.Env:
    """
    Create sinergym environment

    :param env_config: Configuration kwargs for sinergym. Currently, there is only a single key
     in this dictionary, "name". This sets the name of the environment.

    :return: A configured gym environment.
    """

    if not env_config:
        env_config = {"name": "Eplus-5Zone-hot-discrete-v1"}

    env = gym.make(env_config["name"])

    # Taken from https://github.com/ugr-sail/sinergym/blob/main/scripts/DRL_battery.py
    if "normalize" in env_config and env_config["normalize"] is True:
        env_type = env_config["name"].split("-")[1]
        if env_type == "datacenter":
            ranges = RANGES_DATACENTER
        elif env_type == "5Zone":
            ranges = RANGES_5ZONE
        elif env_type == "warehouse":
            ranges = RANGES_WAREHOUSE
        elif env_type == "office":
            ranges = RANGES_OFFICE
        elif env_type == "officegrid":
            ranges = RANGES_OFFICEGRID
        elif env_type == "shop":
            ranges = RANGES_SHOP
        else:
            raise NameError(f"env_type {env_type} is not valid, check environment name")
        env = NormalizeObservation(env, ranges=ranges)

    if "multi_observation" in env_config and env_config["multi_observation"] is True:
        env = MultiObsWrapper(env)

    return env
