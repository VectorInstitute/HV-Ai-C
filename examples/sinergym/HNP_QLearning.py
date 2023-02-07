import gym
from sinergym.utils.rewards import *
import logging
import argparse

import numpy as np
from sinergym.envs import EplusEnv
import wandb
from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
    RANGES_OFFICEGRID,
    RANGES_SHOP
)

from hnp.hnp import HNP


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

    def observation(self, observation):
        """
        Remove the unused state variables from observation

        :param observation: Full observation

        :return: Filtered observation
        """
        return observation[self.obs_to_keep]


logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()
logger.setLevel("INFO")


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


class QLearningAgent:
    """
    Q-Learning Agent Class
    """

    def __init__(
        self,
        env: EplusEnv,
        config: dict,
        obs_mask: np.ndarray,
        use_hnp: bool = True,
    ) -> None:
        """
        Constructor for Q-Learning agent

        :param env: Gym environment
        :param config: Agent configuration
        :param obs_mask: Mask to categorize variables into slowly-changing cont, fast-changing cont, and discrete vars
        :param use_hnp: Enable HNP

        :return: None
        """
        # 3 types --> slowly-changing cont, fast-changing cont, discrete observations
        # actions --> always discrete
        # ASSUMES DISCRETE ACTION SPACE

        self.env = env
        self.config = config
        self.gamma = config["gamma"]
        self.epsilon = config["initial_epsilon"]
        self.epsilon_annealing = config["epsilon_annealing"]
        self.learning_rate = config["learning_rate"]
        self.learning_rate_annealing = config["lr_annealing"]
        self.n_tiles = config["num_tiles"]
        self.use_hnp = use_hnp
        self.reward_type = config["reward_type"]

        # Indices of continuous vars
        self.continuous_idx = np.where(obs_mask <= 1)[0]
        # Indices of discrete vars
        self.discrete_idx = np.where(obs_mask == 2)[0]
        # Reorganized indices of vars: continuous, discrete
        self.permutation_idx = np.hstack(
            (self.continuous_idx, self.discrete_idx))

        # The lower and upper bounds for continuous vars
        self.cont_low = np.zeros(len(self.continuous_idx))
        self.cont_high = np.ones(len(self.continuous_idx))

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()
        self.qtb = np.zeros((*self.obs_space_shape, self.act_space_shape))
        self.vtb = np.zeros(self.obs_space_shape)

        self.state_visitation = np.zeros(self.vtb.shape)

        if self.use_hnp:
            self.hnp = HNP(np.where(obs_mask == 0)[0])

    def get_obs_shape(self) -> tuple:
        """
        Get the observation space shape

        :return: Tuple of discretized observation space for continuous vars and the observation space for discrete vars
        """
        tile_size = 1 / self.n_tiles
        tile_coded_space = [
            np.arange(0, 1 + tile_size, tile_size) for _ in self.continuous_idx
        ]

        return tuple(
            list(list(len(tiles) for tiles in tile_coded_space))
            + list(self.env.observation_space.high[self.discrete_idx])
        )

    def get_act_shape(self) -> int:
        """
        Get the action space shape

        :return: Action space shape
        """
        return self.env.action_space.n

    def choose_action(self, obs_index: np.ndarray, mode: str = "explore") -> int:
        """
        Get action following epsilon-greedy policy

        :param obs_index: Observation index
        :param mode: Training or evaluation

        :return: Action
        """
        if mode == "explore":
            if np.random.rand(1) < self.epsilon:
                return self.env.action_space.sample()
            return np.argmax(self.qtb[tuple(obs_index)])

        if mode == "greedy":  # For evaluation purposes
            return np.argmax(self.qtb[tuple(obs_index)])

    def get_vtb_idx_from_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the value table index from observation

        :param obs: Observation

        :return: Value table index of observation and continuous observation var indices
        """
        obs = obs[self.permutation_idx]
        cont_obs = obs[: len(self.continuous_idx)]

        cont_obs_index_floats = (
            (cont_obs - self.cont_low)
            / (self.cont_high - self.cont_low)
            * (np.array(self.vtb.shape[: len(self.cont_high)]) - 1)
        )
        cont_obs_index = np.round(cont_obs_index_floats)
        obs_index = np.hstack((cont_obs_index, obs[len(self.continuous_idx):])).astype(
            int
        )

        return obs_index, cont_obs_index_floats

    def get_next_value(self, obs: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the new state value

        :param obs: Observation

        :return: Next state value and value table index of observation
        """
        full_obs_index, cont_obs_index_floats = self.get_vtb_idx_from_obs(obs)
        next_value = self.vtb[tuple(full_obs_index)]

        if self.use_hnp:
            next_value = self.hnp.get_next_value(
                self.vtb, full_obs_index, cont_obs_index_floats
            )

        return next_value, full_obs_index

    def train(self) -> None:
        """Q-Learning agent training"""

        # n people, outdoor temperature, indoor temperature
        obs = self.env.reset()
        prev_vtb_index, _ = self.get_vtb_idx_from_obs(obs)
        episodes_reward = []
        episodes_timesteps = []
        ep_n = 0
        total_timesteps = 0
        while ep_n < self.config["num_episodes"]:
            episode_reward = 0
            timesteps = 0
            while True:
                action = self.choose_action(prev_vtb_index)
                # Set value table to value of max action at that state
                self.vtb = np.nanmax(self.qtb, -1)
                obs, rew, done, _ = self.env.step(action)
                episode_reward += rew
                next_value, next_vtb_index = self.get_next_value(obs)

                # Do Q learning update
                prev_qtb_index = tuple([*prev_vtb_index, action])
                self.state_visitation[prev_qtb_index[:-1]] += 1
                curr_q = self.qtb[prev_qtb_index]
                q_target = rew + self.gamma * next_value
                self.qtb[prev_qtb_index] = curr_q + \
                    self.learning_rate * (q_target - curr_q)
                total_timesteps += 1
                timesteps += 1
                prev_vtb_index = next_vtb_index

                if wandb.run:
                    wandb.log({"rollout/timestep_rew": rew})

                if done:
                    episodes_reward.append(episode_reward)
                    episodes_timesteps.append(timesteps)
                    obs = self.env.reset()

                    # Annealing
                    self.epsilon = self.epsilon * self.epsilon_annealing
                    self.learning_rate = self.learning_rate * self.learning_rate_annealing

                    ep_n += 1

                    mean_episodes_return = np.mean(episodes_reward)
                    mean_episodes_timesteps = np.mean(episodes_timesteps)
                    # Logging at the end of each episode
                    if wandb.run:
                        wandb.log({"rollout/ep_rew": episode_reward,
                                   "rollout/ep_rew_mean": mean_episodes_return, "rollout/ep_len_mean": mean_episodes_timesteps, "rollout/exploration_rate": self.epsilon, "train/learning_rate": self.learning_rate})
                    logger.info(
                        f"------------------------\nepisode: {ep_n}\nepisode_return: {episode_reward}\nepisode_timesteps: {timesteps}\naverage_episodes_return: {mean_episodes_return}\naverage_episodes_timesteps: {mean_episodes_timesteps}\ntotal_timesteps: {total_timesteps}\n-------------------------")
                    break


parser = argparse.ArgumentParser(prog="HNP_QLearning")

parser.add_argument("-n", "--num_epsidoes", type=int, default=10)
parser.add_argument("--hnp", action="store_true", default=True)
parser.add_argument("--wandb", action="store_true", default=False)

args = parser.parse_args()

config = {
    "env_name": "Eplus-5Zone-hot-discrete-v1",
    "learning_rate": 0.1,
    "num_episodes": args.num_episodes,
    "initial_epsilon": 0.999,
    "gamma": 0.99,
    "use_hnp": args.hnp,
    "agent": "HNP-QLearning",
    "reward_type": LinearReward,
    "lr_annealing": 1,
    "epsilon_annealing": 1,
    "num_tiles": 20,
    "filter_obs": [4, 5, 13],
    "normalize": True
}

exp_name = f"{'HNP' if args.hnp else ''}_{config['agent']}_{config['env_name']}_{datetime.now():%Y-%m-%d %H:%M:%S}"

if args.wandb:
    # Wandb configuration
    wandb_run = wandb.init(name=exp_name, project="hnp", entity="vector-institute-aieng",
                           config=config)

env = create_env(config)
env = FilterObservation(env, np.array(config['filter_obs']))

agent = QLearningAgent(env, config, np.array([0, 0, 0]), True)

agent.train()

if wandb.run:
    wandb_run.finish()

env.close()
