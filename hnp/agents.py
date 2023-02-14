"""
This file includes all agent classes
"""

import os
import logging
from datetime import datetime, date
from abc import ABC, abstractmethod

import numpy as np
from sinergym.envs import EplusEnv

from hnp.hnp import HNP

logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()
logger.setLevel("INFO")


class Agent(ABC):
    """
    Parent Reinforcement Learning Agent Class
    """

    def __init__(
        self,
        env: EplusEnv,
        config: dict,
        results_dir: str = "training_results",
        use_beobench: bool = False,
    ) -> None:
        """
        Constructor for RL agent

        :param env: Gym environment
        :param config: Agent configuration
        :param results_dir: Directory to save results
        :param use_beobench: Enable beobench

        :return: None
        """
        self.env = env
        self.config = config
        self.results_dir = results_dir
        self.rewards = []
        self.use_beobench = use_beobench

    @abstractmethod
    def train(self) -> None:
        """
        RL agent training
        """

    def save_results(self) -> None:
        """
        Saves training result
        """

        today = date.today()
        day = today.strftime("%Y_%b_%d")
        now = datetime.now()
        time = now.strftime("%H_%M_%S")
        base_dir = "root" if self.use_beobench else os.getcwd()
        dir_name = f"/{base_dir}/{self.results_dir}/{day}/results_{time}"
        os.makedirs(dir_name)

        logging.info("Saving results...")

        np.save(f"{dir_name}/rewards.npy", self.rewards)


class RandomActionAgent(Agent):
    """
    Random Action Agent Class
    """

    def train(self) -> None:
        """
        Random Action agent training
        """

        self.env.reset()
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= self.config["num_episodes"]:
            # Set value table to value of max action at that state

            action = self.env.action_space.sample()
            _, rew, done, _ = self.env.step(action)
            episode_reward += rew

            n_steps += 1
            if n_steps == self.config["horizon"]:  # New episode
                self.rewards.append(episode_reward)
                logger.info("Episode %d --- Reward: %d", ep_n, episode_reward)

                n_steps = 0
                ep_n += 1
                episode_reward = 0

            if done:
                self.env.reset()


class FixedActionAgent(Agent):
    """
    Fixed Action Agent Class
    """

    def train(self) -> None:
        """
        Fixed Action agent training
        """

        self.env.reset()
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= self.config["num_episodes"]:
            # Set value table to value of max action at that state

            _, reward, done, _ = self.env.step(self.config["action_index"])
            episode_reward += reward

            n_steps += 1
            if n_steps == self.config["horizon"]:  # New episode
                self.rewards.append(episode_reward)
                logger.info("Episode %d --- Reward: %d", ep_n, episode_reward)

                n_steps = 0
                ep_n += 1
                episode_reward = 0

            if done:
                self.env.reset()


class QLearningAgent(Agent):
    """
    Q-Learning Agent Class
    """

    def __init__(
        self,
        env: EplusEnv,
        config: dict,
        obs_mask: np.ndarray,
        results_dir: str = "training_results",
        use_beobench: bool = False,
        use_hnp: bool = True,
    ) -> None:
        """
        Constructor for Q-Learning agent

        :param env: Gym environment
        :param config: Agent configuration
        :param obs_mask: Mask to categorize variables into slowly-changing continuous,
         fast-changing continuous, and discrete variables
        :param results_dir: Directory to save results
        :param use_beobench: Enable Beobench
        :param use_hnp: Enable HNP

        :return: None
        """
        # 3 types --> slowly-changing cont, fast-changing cont, discrete observations
        # actions --> always discrete
        # ASSUMES DISCRETE ACTION SPACE
        super().__init__(env, config, results_dir, use_beobench)

        self.gamma = config["gamma"]
        self.epsilon = config["initial_epsilon"]
        self.epsilon_annealing = config["epsilon_annealing"]
        self.learning_rate = config["learning_rate"]
        self.learning_rate_annealing = config["learning_rate_annealing"]
        self.use_hnp = use_hnp

        # Indices of continuous vars
        self.continuous_idx = np.where(obs_mask <= 1)[0]
        # Indices of discrete vars
        self.discrete_idx = np.where(obs_mask == 2)[0]
        # Reorganized indices of vars: continuous, discrete
        self.permutation_idx = np.hstack((self.continuous_idx, self.discrete_idx))

        if type(config["num_tiles"]) is list:
            self.n_tiles = np.array(config["num_tiles"])
        else:
            self.n_tiles = np.full(self.continuous_idx.shape, config["num_tiles"])

        # The lower and upper bounds for continuous vars
        self.cont_low = np.zeros(len(self.continuous_idx))
        self.cont_high = np.ones(len(self.continuous_idx))

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()
        self.qtb = np.zeros((*self.obs_space_shape, self.act_space_shape))
        self.vtb = np.zeros(self.obs_space_shape)

        self.state_visitation = np.zeros(self.vtb.shape)
        self.average_rewards = []

        if self.use_hnp:
            self.hnp = HNP(np.where(obs_mask == 0)[0])

    def get_obs_shape(self) -> tuple:
        """
        Get the observation space shape 
        
        The state space for continuous variables are tile coded based on the number of tiles set
        in the configuration file.

        :return: Tuple of discretized observation space for continuous variables and the observation
         space for discrete variables
        """
        tile_size = 1 / self.n_tiles
        tile_coded_space = [
            np.arange(0, 1 + tile_size[i], tile_size[i]) for i in self.continuous_idx
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

        For continuous variables, first get the indices as floats, then round to integers and stack
        them with discrete variable indices

        :param obs: Observation

        :return: Value table index and continuous variable indices as floats
        """
        obs = obs[self.permutation_idx]
        cont_obs = obs[: len(self.continuous_idx)]

        cont_obs_index_floats = (
            (cont_obs - self.cont_low)
            / (self.cont_high - self.cont_low)
            * (np.array(self.vtb.shape[: len(self.cont_high)]) - 1)
        )
        cont_obs_index = np.round(cont_obs_index_floats)
        obs_index = np.hstack((cont_obs_index, obs[len(self.continuous_idx) :])).astype(
            int
        )

        return obs_index, cont_obs_index_floats

    def get_next_value(self, obs: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the new state value

        If not using HNP, the new state value is retrieved from the value table directly.
        For HNP computation, refer to the get_next_value function in HNP class.

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
        """
        Q-Learning agent training

        The training follows conventional Q-Learning update rule. If the episode length is less than
        the total number of weather data points, then the environment would not reset by the end of
        the episode. The environment only resets when all the weather data points have been used up.
        """

        # n people, outdoor temperature, indoor temperature
        obs = self.env.reset()
        prev_vtb_index, _ = self.get_vtb_idx_from_obs(obs)
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n < self.config["num_episodes"]:
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
            self.qtb[prev_qtb_index] = curr_q + self.learning_rate * (q_target - curr_q)
            n_steps += 1
            prev_vtb_index = next_vtb_index
            if n_steps == self.config["horizon"]:  # New episode
                if ep_n % 10 == 0:
                    logger.info(
                        "Episode %d --- Reward: %d Average reward per timestep: %.2f",
                        ep_n,
                        episode_reward,
                        (episode_reward / n_steps),
                    )
                avg_reward = episode_reward / n_steps
                self.rewards.append(episode_reward)
                self.average_rewards.append(avg_reward)

                n_steps = 0
                ep_n += 1
                episode_reward = 0
                self.epsilon = self.epsilon * self.epsilon_annealing
                self.learning_rate = self.learning_rate * self.learning_rate_annealing
            if done:
                obs = self.env.reset()
