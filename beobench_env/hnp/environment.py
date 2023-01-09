import numpy as np
import gym
from gym import spaces

from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_to_keep, lows, highs, mask):
        super().__init__(env)
        self.env = env
        self.obs_to_keep = obs_to_keep
        self.lows = lows
        self.highs = highs
        self.mask = mask
        self.observation_space = spaces.Box(
            low=lows,
            high=highs,
            shape=((len(obs_to_keep),)),
            dtype=self.env.observation_space.dtype,
        )

    def observation(self, obs):
        if np.max(obs) > 1:
            print("more than 0")
        if np.min(obs) < 0:
            print("less 0 ")
        # modify obs
        return np.clip(obs[self.obs_to_keep], self.lows, self.highs)


def create_env(env_config: dict = None) -> gym.Env:
    """Create sinergym environment.
    Args:
        env_config (dict, optional): configuration kwargs for sinergym. Currently,
            there is only a single key in this dictionary, "name". This sets
            the name of the environment. Defaults to None.
    Returns:
        gym.Env: a configured gym environment.
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
