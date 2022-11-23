import numpy as np
import gym
from gym import spaces
from gym.spaces import Box, Discrete, Tuple, Dict
import time
import os
CONSTANT_NINF = -9e99

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_to_keep, lows, highs, mask):
        super().__init__(env)
        self.env = env
        self.obs_to_keep = obs_to_keep
        self.lows = lows
        self.highs = highs
        self.mask = mask
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
        obs_mask,
        low,
        high,
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
        # 3 types --> discrete, continuous, slow continuous observations
        # actions --> always discrete
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
        self.low = low
        self.high = high

        self.slow_continuous_idx = np.where(obs_mask==0)[0]
        self.fast_continuous_idx = np.where(obs_mask==1)[0]
        self.to_discretize_idx = np.hstack((self.slow_continuous_idx, self.fast_continuous_idx))# Slow first

        self.cont_low = self.low[self.to_discretize_idx]
        self.cont_high = self.high[self.to_discretize_idx]
        self.discrete_idx = np.where(obs_mask==2)[0]
        self.permutation_idx = np.hstack((self.slow_continuous_idx, self.fast_continuous_idx, self.discrete_idx))
        self.n_slow_cont = len(self.slow_continuous_idx)
        self.n_fast_discrete = len(self.fast_continuous_idx) + len(self.discrete_idx)

        self.obs_steps = np.ones(len(self.low)) /20 # TODO: CHANGE THIS
        self.discretization_ticks = self.get_ticks(self.env.observation_space, self.obs_steps) 

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()
        self.qtb = self.make_qtb()
        
        self.vtb = self.make_vtb()
        self.state_visitation = np.zeros(self.vtb.shape)
        self.rewards = []
        self.average_rewards = []

        n_dim = len(self.obs_space_shape)
        if self.n_slow_cont > 0:
            portion_index_matrix = np.vstack((np.zeros(self.n_slow_cont), np.ones(self.n_slow_cont))).T
            self.all_portion_index_combos = np.array(np.meshgrid(*portion_index_matrix), dtype=int).T.reshape(-1, self.n_slow_cont)

    def transform_obs(self, obs):
        return obs[self.permutation_idx]

    def make_qtb(self):
        return np.zeros((*self.obs_space_shape, self.act_space_shape))
        
    def make_vtb(self):
        return np.zeros(self.obs_space_shape)

    def get_obs_shape(self):
        return tuple(list([len(ticks) for ticks in self.discretization_ticks]) + list(self.high[self.discrete_idx]))

    def get_act_shape(self):
        return self.env.action_space.n

    def get_ticks(self, space, steps):
        return [np.arange(space.low[i], space.high[i] + steps[i], steps[i]) for i in self.to_discretize_idx]

    def obs_to_index_float(self, obs):
        return (obs - self.cont_low)/(self.cont_high - self.cont_low) * (np.array(self.vtb.shape[:len(self.cont_high)]) - 1)
    
    def choose_action(self, obs_index, mode="explore"):
        if mode == "explore":
            if np.random.rand(1) < self.eps:
                return self.env.action_space.sample()
            return np.argmax(self.qtb[tuple(obs_index)])
        
        if mode == "greedy": # For evaluation purposes
            return np.argmax(self.qtb[tuple(obs_index)])

    def get_vtb_idx_from_obs(self, obs):
        obs = self.transform_obs(obs)
        cont_obs = obs[:len(self.to_discretize_idx)]

        cont_obs_index_floats = self.obs_to_index_float(cont_obs)
        cont_obs_index = np.round(cont_obs_index_floats)
        obs_index = np.hstack((cont_obs_index, obs[len(self.to_discretize_idx):])).astype(int)

        return obs_index, cont_obs_index_floats


    def get_next_value(self, obs):
        # If change first 5 lines of this function also
        full_obs_index, cont_obs_index_floats = self.get_vtb_idx_from_obs(obs)
        if self.n_slow_cont == 0: # No HNP calculation needed
            return self.vtb[tuple(full_obs_index)], full_obs_index
        slow_cont_obs_index_floats = cont_obs_index_floats[:len(self.slow_continuous_idx)]
        slow_cont_obs_index_int_below = np.floor(slow_cont_obs_index_floats).astype(np.int32)
        slow_cont_obs_index_int_above = slow_cont_obs_index_int_below + 1

        if len(self.to_discretize_idx) < len(obs):
            discrete_obs = obs[len(self.to_discretize_idx) + 1:]

        vtb_index_matrix = np.vstack((slow_cont_obs_index_int_below, slow_cont_obs_index_int_above)).T
        all_vtb_index_combos = np.array(np.meshgrid(*vtb_index_matrix)).T.reshape(-1, len(slow_cont_obs_index_int_above))

        portion_below = slow_cont_obs_index_int_above - slow_cont_obs_index_floats
        portion_above = 1 - portion_below
        portion_matrix = np.vstack((portion_below, portion_above)).T

        non_hnp_index = full_obs_index[len(self.slow_continuous_idx):]
        next_value = 0
        for i, combo in enumerate(self.all_portion_index_combos):
            portions = portion_matrix[np.arange(len(slow_cont_obs_index_floats)), combo]
            value_from_vtb = self.vtb[tuple(np.hstack((all_vtb_index_combos[i], non_hnp_index)).astype(int))]
            next_value += np.prod(portions) * value_from_vtb
        
        return next_value, full_obs_index
    
    def learn(self, n_episodes, horizon) -> None:
        # n people, outdoor temperature, indoor temperature
        obs = self.env.reset()
        prev_vtb_index, _ = self.get_vtb_idx_from_obs(obs)
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= n_episodes: 
            ac = self.choose_action(prev_vtb_index)
            # Set value table to value of max action at that state
            self.vtb = np.nanmax(self.qtb, -1)
            obs, rew, done, info = self.env.step(ac)
            episode_reward += rew
            next_value, next_vtb_index = self.get_next_value(obs)

            # Do Q learning update
            prev_qtb_index = tuple([*prev_vtb_index, ac])
            self.state_visitation[prev_qtb_index[:-1]] += 1
            curr_q = self.qtb[prev_qtb_index]
            q_target = rew + self.gamma * next_value
            self.qtb[prev_qtb_index] = curr_q + self.learning_rate * (q_target - curr_q)
            n_steps += 1
            prev_vtb_index = next_vtb_index
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
                obs = self.env.reset()
    
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
obs_to_keep = np.array([6,7,8]) 
# lows = np.array([0,0,0,0,0,0])
# highs = np.array([1,1,1,8,1,9]) # if discrete then is number of actions
# mask = np.array([0,1,0,2,1,2]) # 0 = slow moving continuous, 1 = fast moving continuous, 2 = discrete
lows = np.array([0,0,0])
highs = np.array([5,1,1]) # if discrete then is number of actions
mask = np.array([2,0,1]) 
env = create_env()

env = ObservationWrapper(env, obs_to_keep, lows, highs, mask)

agent = Agent(
    env, 
    mask,
    lows,
    highs,
    eps_annealing=config["hyperparams"]["eps_annealing"], 
    learning_rate_annealing=config["hyperparams"]["lr_annealing"])

exit()
agent.learn(config["agent"]["config"]["num_episodes"], config["hyperparams"]["horizon"])
agent.save_results()
env.close()

