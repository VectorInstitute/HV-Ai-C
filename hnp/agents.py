import os
import logging
from datetime import datetime, date
from abc import ABC, abstractmethod

import numpy as np

from hnp.hnp import HNP


class Agent(ABC):
    """
    Parent Reinforcement Learning Agent Class 
    """

    def __init__(self, env, config) -> None:
        """
        Constructor for RL agent

        Args:
            env: Gym environment
            config: Agent configuration

        Returns:
            None
        """
        self.env = env
        self.config = config
        self.rewards = []

    @abstractmethod
    def train(self) -> None:
        """
        RL agent training

        Args:
            None
        
        Returns:
            None
        """
        pass

    def save_results(self) -> None:
        """
        Saves training result

        Args:
            None

        Returns:
            None
        """
        today = date.today()
        day = today.strftime("%Y_%b_%d")
        now = datetime.now()
        time = now.strftime("%H_%M_%S")
        dir_name = f"{os.getcwd()}/beobench_results/{day}/results_{time}"
        os.makedirs(dir_name)

        logging.info("Saving results...")

        np.save(f"{dir_name}/rewards.npy", self.rewards)


class RandomActionAgent(Agent):
    """
    Random Action Agent Class
    """

    def __init__(self, env, config) -> None:
        """
        Constructor for Random Action agent

        Args:
            env: Gym environment
            config: Agent configuration

        Returns:
            None
        """
        super().__init__(env, config)

    def train(self) -> None:
        """
        Random Action agent training

        Args:
            None
        
        Returns:
            None
        """
        self.env.reset()
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= self.config["num_episodes"]:
            # Set value table to value of max action at that state

            ac = self.env.action_space.sample()
            _, rew, done, _ = self.env.step(ac)
            episode_reward += rew

            n_steps += 1
            if n_steps == self.config["horizon"]:  # New episode
                self.rewards.append(episode_reward)
                logging.info(f"Episode {ep_n} --- Reward: {episode_reward}")

                n_steps = 0
                ep_n += 1
                episode_reward = 0

            if done:
                self.env.reset()


class FixedActionAgent(Agent):
    """
    Fixed Action Agent Class
    """

    def __init__(self, env, config) -> None:
        """
        Constructor for Fixed Action agent

        Args:
            env: Gym environment
            config: Agent configuration

        Returns:
            None
        """

        super().__init__(env, config)

    def train(self) -> None:
        """
        Fixed Action agent training

        Args:
            None
        
        Returns:
            None
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
                logging.info(f"Episode {ep_n} --- Reward: {episode_reward}")

                n_steps = 0
                ep_n += 1
                episode_reward = 0

            if done:
                self.env.reset()


class QLearningAgent(Agent):
    """
    Q-Learning Agent Class
    """

    def __init__(self, env, config, obs_mask, low, high, use_hnp=True) -> None:
        """
        Constructor for Q-Learning agent

        Args:
            env: Gym environment
            config: Agent configuration
            obs_mask: Mask to categorize variables into slowly-changing cont, fast-changing cont, and discrete vars
            low: Lower bound of the state variables
            high: Upper bound of the state variables
            use_hnp: Whether to use HNP

        Returns:
            None
        """
        # 3 types --> slowly-changing cont, fast-changing cont, discrete observations
        # actions --> always discrete
        # ASSUMES DISCRETE ACTION SPACE
        super().__init__(env, config)

        self.gamma = config["gamma"]
        self.epsilon = config["initial_epsilon"]
        self.epsilon_annealing = config["epsilon_annealing"]
        self.learning_rate = config["learning_rate"]
        self.learning_rate_annealing = config["learning_rate_annealing"]
        self.step_size = config["step_size"]
        self.low = low
        self.high = high
        self.use_hnp = use_hnp

        # Indices of continuous vars
        self.continuous_idx = np.where(obs_mask <= 1)[0]
        # Indices of discrete vars
        self.discrete_idx = np.where(obs_mask == 2)[0]
        # Reorganized indices of vars: continuous, discrete
        self.permutation_idx = np.hstack(
            (self.continuous_idx, self.discrete_idx)
        )

        # The lower and upper bounds for continuous vars
        self.cont_low = self.low[self.continuous_idx]
        self.cont_high = self.high[self.continuous_idx]

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()
        self.qtb = np.zeros((*self.obs_space_shape, self.act_space_shape))
        self.vtb = np.zeros(self.obs_space_shape)

        self.state_visitation = np.zeros(self.vtb.shape)
        self.average_rewards = []

        if self.use_hnp:
            self.hnp = HNP(np.where(obs_mask == 0)[0])

    def get_obs_shape(self):
        """
        Get the observation space shape

        Args:
            None

        Returns:
            Tuple of discretized observation space for continuous vars and
            the observation space for discrete vars
        """
        # TODO: SIMPLIFY THIS
        steps = np.ones(len(self.low)) / 20
        discretization_ticks = [
            np.arange(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i] + steps[i],
                steps[i],
            )
            for i in self.continuous_idx
        ]

        return tuple(
            list(list(len(ticks) for ticks in discretization_ticks))
            + list(self.high[self.discrete_idx])
        )

    def get_act_shape(self):
        """
        Get the action space shape

        Args:
            None

        Returns:
            Action space shape
        """
        return self.env.action_space.n

    def choose_action(self, obs_index, mode="explore"):
        """
        Get action following epsilon-greedy policy

        Args:
            obs_index: Observation index
            mode: Training or evaluation

        Returns:
            Action
        """
        if mode == "explore":
            if np.random.rand(1) < self.epsilon:
                return self.env.action_space.sample()
            return np.argmax(self.qtb[tuple(obs_index)])

        if mode == "greedy":  # For evaluation purposes
            return np.argmax(self.qtb[tuple(obs_index)])

    def get_vtb_idx_from_obs(self, obs):
        """
        Get the value table index from observation

        Args:
            obs: Observation

        Returns:
            obs_index: Value table index of observation
            cont_obs_index_floats: Continuous obseravation var indices
        """
        obs = obs[self.permutation_idx]
        cont_obs = obs[: len(self.continuous_idx)]

        cont_obs_index_floats = (
            (cont_obs - self.cont_low)
            / (self.cont_high - self.cont_low)
            * (np.array(self.vtb.shape[: len(self.cont_high)]) - 1)
        )
        cont_obs_index = np.round(cont_obs_index_floats)
        obs_index = np.hstack(
            (cont_obs_index, obs[len(self.continuous_idx) :])
        ).astype(int)

        return obs_index, cont_obs_index_floats

    def get_next_value(self, obs):
        """
        Computes the new state value

        Args:
            obs: Observation
        
        Returns:
            next_value: Next state value
            full_obs_index: Value table index of observation
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

        Args:
            None
        
        Returns:
            None
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
                print(f"num_timesteps: {n_steps}")
                print(
                    f"Episode {ep_n} --- Reward: {episode_reward}, \
                    Average reward per timestep: {episode_reward/n_steps}"
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
