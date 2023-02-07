from sinergym.utils.rewards import *
import gym
from gym.spaces import Box
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
import wandb
from wandb.integration.sb3 import WandbCallback
from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
    RANGES_OFFICEGRID,
    RANGES_SHOP
)


def create_env(env_config: dict = None) -> gym.Env:
    """
    Create sinergym environment

    :param env_config: Configuration kwargs for sinergym. Currently, there is only a single key
     in this dictionary, "name". This sets the name of the environment.

    :return: A configured gym environment.
    """

    if not env_config:
        env_config = {"env_name": "Eplus-5Zone-hot-discrete-v1"}

    env = gym.make(env_config["env_name"], reward=config["reward_type"])

    # Taken from https://github.com/ugr-sail/sinergym/blob/main/scripts/DRL_battery.py
    if "normalize" in env_config and env_config["normalize"] is True:
        env_type = env_config["env_name"].split("-")[1]
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
            raise NameError(
                f"env_type {env_type} is not valid, check environment name")
        env = NormalizeObservation(env, ranges=ranges)

    if "multi_observation" in env_config and env_config["multi_observation"] is True:
        env = MultiObsWrapper(env)

    return env


class FilterObservation(gym.ObservationWrapper):
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
        unwrapped_obs_space = self.observation_space
        self.observation_space = Box(
            unwrapped_obs_space.low[self.obs_to_keep], unwrapped_obs_space.high[self.obs_to_keep], shape=(len(self.obs_to_keep), ))

    def observation(self, observation):
        """
        Remove the unused state variables from observation

        :param observation: Full observation

        :return: Filtered observation
        """
        return observation[self.obs_to_keep]


config = {
    "env_name": "Eplus-5Zone-hot-discrete-v1",
    "learning_rate": 0.1,
    "num_episodes": 10,
    "initial_epsilon": 0.999,
    "gamma": 0.99,
    "use_hnp": False,
    "agent": "DQN",
    "policy_type": "MlpPolicy",
    "reward_type": LinearReward,
    # "lr_annealing": 0.999,
    # "epsilon_annealing": 0.999,
    "obs_filter": [4, 5, 13],
    "normalize": True
}
exp_name = f"{config['agent']}_{config['env_name']}_{datetime.now():%Y-%m-%d %H:%M:%S}"

# Wandb initialization
run = wandb.init(project="hnp", entity="vector-institute-aieng",
                 config=config, sync_tensorboard=True, monitor_gym=False, name=exp_name)


env = create_env(config)
env = FilterObservation(env, np.array(config["obs_filter"]))
env = Monitor(env)

n_timesteps_episode = env.simulator._eplus_one_epi_len / \
    env.simulator._eplus_run_stepsize

vec_env = DummyVecEnv([lambda: env])

model = DQN(config["policy_type"], vec_env, verbose=1,
            tensorboard_log=f"runs/{run.id}", learning_rate=config["learning_rate"], exploration_final_eps=config['initial_epsilon'], exploration_initial_eps=config["initial_epsilon"])

total_timesteps = config["num_episodes"] * n_timesteps_episode

model.learn(total_timesteps=total_timesteps, callback=WandbCallback(
    gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2), log_interval=1)

run.finish()

env.close()
