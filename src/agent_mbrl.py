import numpy as np
import gym
from gym import spaces
import time
CONSTANT_NINF = -9e99

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_to_keep, lows, highs):
        super().__init__(env)
        self.env = env
        self.obs_to_keep = obs_to_keep
        self.lows = lows
        self.highs = highs
        self.observation_space = \
            spaces.Box(
                low=lows, 
                high=highs, 
                shape=((len(obs_to_keep),)), 
                dtype=self.env.observation_space.dtype
                )
    
    def observation(self, obs):
        if np.max(obs) > 1:
            print("more than 0")
        if np.min(obs) < 0:
            print("less 0 ")
        # modify obs
        return np.clip(obs[self.obs_to_keep], self.lows, self.highs)

from beobench.experiment.provider import create_env, config
import numpy as np
import os
import time
import gym
import itertools
import datetime
import pickle
import os
import logging

from numpy.lib import recfunctions
import pandas as pd

class Agent:
    """
    Reinforcement Learning Agent Class for learning the policy and the model
    """

    def __init__(
        self,
        env,
        initial_eps =  1,
        eps_annealing = 0.9,
        eps_annealing_interval = 100000,
        learning_rate: float = 0.1,
        learning_rate_annealing: float = 0.9,
        temp_step: float = 1.0,
        gamma: float = 0.99,
        theta: float = 1e-6,
        n_iteration_long: int = 1000,
    ) -> None:
        """
        """
        # ASSUMES CONTINUOUS STATE SPACE AND DISCRETE ACTION SPACE
        self.env = env

        self.temp_step = temp_step
        self.gamma = gamma
        self.theta = theta
        self.n_iteration_long = n_iteration_long
        self.eps = initial_eps
        self.eps_annealing = eps_annealing
        self.eps_annealing_interval = eps_annealing_interval
        self.learning_rate = learning_rate
        self.learning_rate_annealing = learning_rate_annealing

        self.qtb = None
        self.vtb = None

        self.obs_steps = np.ones(self.env.observation_space.shape) /20
        self.obs_ticks = self.get_ticks(self.env.observation_space, self.obs_steps) 

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()

        self.qtb = self.make_qtb()
        
        self.vtb = self.make_vtb()
        self.state_visitation = np.zeros(self.vtb.shape)
        self.rewards = []
        self.average_rewards = []

        n_dim = len(self.obs_space_shape)
        print("n dim", n_dim)
        portion_index_matrix = np.vstack((np.zeros(n_dim), np.ones(n_dim))).T
        self.all_portion_index_combos = np.array(np.meshgrid(*portion_index_matrix), dtype=int).T.reshape(-1, n_dim)

    def make_qtb(self):
        return np.zeros((*self.obs_space_shape, self.act_space_shape))
        
    def make_vtb(self):
        return np.zeros(self.obs_space_shape)

    def get_obs_shape(self):
        return tuple([len(ticks) for ticks in self.obs_ticks])

    def get_act_shape(self):
        return self.env.action_space.n

    def get_ticks(self, space, steps):
        return [np.arange(space.high[i], space.low[i] - steps[i], -steps[i]) for i, high in enumerate(space.high)]

    def obs_to_index_float(self, obs):
        return (obs - self.env.observation_space.low)/(self.env.observation_space.high - self.env.observation_space.low) * (np.array(self.qtb.shape[:-1]) - 1)
    
    def choose_action(self, obs_index, mode="explore"):
        if mode == "explore":
            if np.random.rand(1) < self.eps:
                return self.env.action_space.sample()
            return np.argmax(self.qtb[tuple(obs_index)])
        
        if mode == "greedy": # For evaluation purposes
            return np.argmax(self.qtb[tuple(obs_index)])

    def get_next_value(self, obs):
        n_dim = len(obs)

        # Calculate HNP Q Target
        next_obs_index_floats = self.obs_to_index_float(obs)
        next_obs_index_int_below = np.floor(next_obs_index_floats).astype(np.int32)
        next_obs_index_int_above = next_obs_index_int_below + 1
        next_obs_index = np.round(next_obs_index_floats).astype(int)

        vtb_index_matrix = np.vstack((next_obs_index_int_below, next_obs_index_int_above)).T
        all_vtb_index_combos = np.array(np.meshgrid(*vtb_index_matrix)).T.reshape(-1, len(obs))

        portion_below = next_obs_index_int_above - next_obs_index_floats
        portion_above = 1 - portion_below
        portion_matrix = np.vstack((portion_below, portion_above)).T

        next_value = 0
        for i, combo in enumerate(self.all_portion_index_combos):
            portions = portion_matrix[np.arange(n_dim), combo]
            value_from_vtb = self.vtb[tuple(all_vtb_index_combos[i])]
            next_value += np.prod(portions) * value_from_vtb

        return next_value, next_obs_index
    
    def learn(self, n_episodes, horizon) -> None:
        prev_obs = self.env.reset()
        next_obs_index = np.zeros(prev_obs.shape, dtype=int)
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= n_episodes: 
            # Set value table to value of max action at that state
            self.vtb = np.nanmax(self.qtb, -1)
            
            ac = self.choose_action(next_obs_index)
            obs, rew, done, info = self.env.step(ac)
            episode_reward += rew

            next_value, next_obs_index = self.get_next_value(obs)
            # Do Q learning update
            prev_qtb_index = np.round(self.obs_to_index_float(obs)).astype(np.int32)
            prev_qtb_index = tuple([*prev_qtb_index, ac])
            self.state_visitation[prev_qtb_index[:-1]] += 1

            curr_q = self.qtb[prev_qtb_index]
            q_target = rew + self.gamma * np.sum(next_value)
            self.qtb[prev_qtb_index] = curr_q + self.learning_rate * (q_target - curr_q)
            prev_obs = obs
            n_steps += 1
            if done: # New episode
                print(f"num_timesteps: {n_steps}")
                print(f"Episode {ep_n} --- Reward: {episode_reward}, Average reward per timestep: {episode_reward/n_steps}")
                avg_reward = episode_reward/n_steps
                self.rewards.append(episode_reward)
                self.average_rewards.append(avg_reward)

                n_steps = 0
                ep_n += 1
                self.eps = self.eps * self.eps_annealing
                self.learning_rate = self.learning_rate * self.learning_rate_annealing
                episode_reward = 0
                prev_obs = self.env.reset()
    
    def save_results(self):
        today = date.today()
        day = today.strftime("%Y_%b_%d")
        now = dt.now()
        time = now.strftime("%H_%M_%S")
        dir_name = f"/root/beobench_results/{day}/results_{time}"
        os.makedirs(dir_name)

        original_stdout = sys.stdout # Save a reference to the original standard output

        with open(f"{dir_name}/params.json", 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(json.dumps(config, indent=4))
            sys.stdout = original_stdout
        print("Saving results...")

        np.savez(
            f"{dir_name}/metrics.npz", 
            qtb=self.qtb, 
            rewards=self.rewards,
            average_rewards=self.average_rewards,
            state_visitation=self.state_visitation
            )

from beobench.experiment.provider import create_env, config
import numpy as np
import os
import time
import gym

# Create environment and wrap observations
obs_to_keep = np.array([8]) 
lows = np.array([0])
highs = np.array([1])
env = create_env()
env = ObservationWrapper(env, obs_to_keep, lows, highs)

agent = Agent(env)
agent.learn(config["agent"]["config"]["num_episodes"])
agent.save_results(f"results_{time.time()}.npz")
env.close()

