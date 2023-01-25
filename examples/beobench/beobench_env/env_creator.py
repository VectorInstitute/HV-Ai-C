"""Module to create sinergym environments. """

import gym
import sinergym
import sinergym.utils.wrappers

from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
)


def create_env(env_config: dict = None) -> gym.Env:
    """Create sinergym environment.
    Args:
        env_config (dict, optional): configuration kwargs for sinergym. Currently,
            there is only a single key in this dictionary, "name". This sets
            the name of the environment. Defaults to None.
    Returns:
        gym.Env: a configured gym environment.
    """

    if env_config is None:
        env_config = {"name": "Eplus-5Zone-hot-continuous-v1"}

    # importing sinergym automatically nicely registers
    # sinergym envs with OpenAI gym
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
        env = sinergym.utils.wrappers.NormalizeObservation(env, ranges=ranges)

    if "multi_observation" in env_config and env_config["multi_observation"] is True:
        env = sinergym.utils.wrappers.MultiObsWrapper(env)

    return env