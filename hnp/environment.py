"""
This file includes all environment related class and methods
"""

import gym

from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
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
        self.obs_to_keep = obs_to_keep

    def observation(self, observation):
        """
        Remove the unused state variables from observation

        :param observation: Full observation

        :return: Filtered observation
        """
        return observation[self.obs_to_keep]


def create_env(env_config: dict = None) -> gym.Env:
    """Create sinergym environment.

    :param env_config: Configuration kwargs for sinergym. Currently, there is only a single key in this dictionary, "name". This sets the name of the environment.

    :return: A configured gym environment.
    """

    if not env_config:
        env_config = {"name": "Eplus-5Zone-hot-discrete-v1"}

    env = gym.make(env_config["name"])

    # Taken from
    # https://github.com/jajimer/sinergym/blob/24a37965f4e749faf6caaa3d4ece95330a478904/DRL_battery.py#L221
    if "normalize" in env_config and env_config["normalize"] is True:
        # We have to know what dictionary ranges to use
        env_type = env_config["name"].split("-")[1]
        if env_type == "datacenter":
            ranges = RANGES_DATACENTER
        elif env_type == "5Zone":
            ranges = RANGES_5ZONE
        elif env_type == "warehouse":
            ranges = RANGES_WAREHOUSE
        elif env_type == "office":
            ranges = RANGES_OFFICE
        else:
            raise NameError(f"env_type {env_type} is not valid, check environment name")
        env = NormalizeObservation(env, ranges=ranges)

    if "multi_observation" in env_config and env_config["multi_observation"] is True:
        env = MultiObsWrapper(env)

    return env
