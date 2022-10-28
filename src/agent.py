"""
This file contains the Agent class.
"""
import itertools
import datetime
import pickle
import os
import logging

import numpy as np
from numpy.lib import recfunctions
import pandas as pd

from src.util import CONSTANT_NINF, get_dir_path
from src.environment import Environment


class Agent:
    """
    Reinforcement Learning Agent Class for learning the policy and the model
    """

    def __init__(
        self,
        env: Environment,
        temp_step: float = 1.0,
        gamma: float = 0.95,
        theta: float = 1e-6,
        n_iteration_long: int = 1000,
    ) -> None:
        """
        Constructor for RL Agent class

        Args:
            env: Reinforcement learning environment
            config_name (optional): Config file name. Defaults to agent_params.cfg
            temp_step (optional): Degree for the tile in reinforcement learning.
                Don't change. Defaults to 1.0.
            gamma (optional): Discount in reinforcement learning. Defaults to 0.95.
            theta (optional): Convergence limit. Defaults to 1e-6.
            n_iteration_long (optional): Maximum runs for convergent. Defaults to 1000.

        Returns:
            None
        """
        self.env = env

        self.temp_step = temp_step
        self.gamma = gamma
        self.theta = theta
        self.n_iteration_long = n_iteration_long

        self.make_arr_temp_ticks()
        self.make_arr_state_action()

        self.logger = logging.getLogger("root")

        self.qtb = None
        self.vtb = None
        self.arr_delta_q_long = None
        self.arr_max_delta_q_long = None
        self.arr_state_action_long = None
        self.len_arr_max_delta_q_long = None
        self.model_calculation_time = None

        self.ptb = None
        self.pdf = None

    def make_arr_temp_ticks(self) -> None:
        """
        Constructs the possible bin values for temperature based on maximum/minimum and step values
        for the temperature.

        Returns:
            None
        """
        self.arr_temp_ticks = np.arange(
            self.env.temp_max, (self.env.temp_min - self.temp_step), -self.temp_step
        )
        self.grid_size_temp = len(self.arr_temp_ticks)

    def temp_to_index_float(self, temp: float) -> float:
        """
        Returns the index of given temp value from arr_temp_ticks array as a float

        Args:
            temp: Current temperature value

        Returns:
            Index of temp variable as a float in arr_temp_ticks
        """
        return (
            (self.env.temp_max - temp)
            / (self.env.temp_max - self.env.temp_min)
            * (self.grid_size_temp - 1)
        )

    def make_arr_state_action(self) -> None:
        """
        Constructs the state-action space according to all allowed actions and spaces based on
        the given environment.

        Returns:
            None
        """
        ls_state_action_full = list(
            itertools.product(
                enumerate(self.arr_temp_ticks),
                range(self.env.observation_space["hvac"].n),
                range(self.env.action_space.n),
            )
        )
        arr_state_action = np.array(
            [
                (index_temp, temp, hvac, action)
                for (index_temp, temp), hvac, action in ls_state_action_full
            ],
            dtype=[
                ("index_temp", np.int64),
                ("temp", np.float64),
                ("hvac", np.int64),
                ("action", np.int64),
            ],
        )

        arr_allowed_state_actions = np.array(
            [
                [self.env.status_compressor_on, self.env.action_compressor_on],
                [self.env.status_compressor_on, self.env.action_idle],
                [self.env.status_freecool_on, self.env.action_compressor_on],
                [self.env.status_freecool_on, self.env.action_freecool_on],
                [self.env.status_freecool_on, self.env.action_idle],
                [self.env.status_heater_on, self.env.action_heator_on],
                [self.env.status_heater_on, self.env.action_idle],
                [self.env.status_idle, self.env.action_compressor_on],
                [self.env.status_idle, self.env.action_freecool_on],
                [self.env.status_idle, self.env.action_heator_on],
                [self.env.status_idle, self.env.action_idle],
            ]
        )
        arr_filter_state_action = (
            (
                recfunctions.structured_to_unstructured(
                    arr_state_action[["hvac", "action"]]
                )
                == arr_allowed_state_actions[:, None]
            )
            .all(-1)
            .any(0)
        )
        arr_state_action = arr_state_action[arr_filter_state_action]
        self.arr_state_action_general = arr_state_action[
            (
                (
                    (arr_state_action["temp"] < self.env.setpoint_temp_low + 1.0)
                    & (arr_state_action["action"] == self.env.action_heator_on)
                )
                | (
                    (arr_state_action["temp"] > self.env.setpoint_temp_high - 4.0)
                    & (arr_state_action["action"] != self.env.action_heator_on)
                )
            )
            | (arr_state_action["action"] == self.env.action_idle)
        ]

    def dp_long(self, n_iteration_long: int = None) -> None:
        """
        Dynamic programming algorithm for updating Q table

        Args:
            n_iteration_long (optional): Number of iterations. Defaults to None.

        Returns:
            None
        """
        if not n_iteration_long:
            n_iteration_long = self.n_iteration_long

        self.env.temp_outdoor = self.env.temp_outdoor_nominal

        self.qtb = np.full(
            (
                self.grid_size_temp + 1,
                self.env.observation_space["hvac"].n,
                self.env.action_space.n,
            ),
            CONSTANT_NINF,
        )
        self.vtb = np.zeros(
            (self.grid_size_temp + 1, self.env.observation_space["hvac"].n)
        )

        self.arr_delta_q_long = np.zeros((n_iteration_long, np.product(self.qtb.shape)))
        self.arr_max_delta_q_long = np.zeros(n_iteration_long)
        flag_converge = False

        self.arr_state_action_long = self.arr_state_action_general[
            ~(
                (
                    self.arr_state_action_general["temp"]
                    < self.env.temp_outdoor_nominal + 0.5
                )
                & (
                    self.arr_state_action_general["action"]
                    == self.env.action_freecool_on
                )
            )
        ]

        for i_iteration in range(n_iteration_long):
            qtb_new = np.full(
                (
                    self.grid_size_temp + 1,
                    self.env.observation_space["hvac"].n,
                    self.env.action_space.n,
                ),
                CONSTANT_NINF,
            )
            for index_temp, temp, hvac, action in self.arr_state_action_long:
                (
                    next_temp,
                    next_hvac,
                ), reward = self.env.calculate_next_state_and_reward(action, temp, hvac)

                next_index_temp_float = self.temp_to_index_float(next_temp)
                next_index_temp_int_below = int(next_index_temp_float)
                next_index_temp_int_above = next_index_temp_int_below + 1

                portion_below = next_index_temp_int_above - next_index_temp_float
                portion_above = 1 - portion_below

                next_value_below = self.vtb[next_index_temp_int_below, next_hvac]
                next_value_above = self.vtb[next_index_temp_int_above, next_hvac]

                next_value = (
                    next_value_below * portion_below + next_value_above * portion_above
                )

                q_target = reward + self.gamma * next_value
                qtb_new[index_temp, hvac, action] = q_target

            self.arr_delta_q_long[i_iteration] = np.abs(qtb_new - self.qtb).flatten()
            self.arr_max_delta_q_long[i_iteration] = np.nanmax(
                self.arr_delta_q_long[i_iteration]
            )

            self.qtb = qtb_new.copy()
            self.vtb = np.nanmax(qtb_new, -1)

            if self.arr_max_delta_q_long[i_iteration] < self.theta:
                self.logger.info(
                    "****** Long training converged at iteration %d ********",
                    i_iteration,
                )
                flag_converge = True
                self.arr_delta_q_long = self.arr_delta_q_long[:i_iteration].copy()
                self.arr_max_delta_q_long = self.arr_max_delta_q_long[
                    :i_iteration
                ].copy()
                break

        self.len_arr_max_delta_q_long = i_iteration + 1
        if not flag_converge:
            self.logger.warning(
                "*****************************************************************\n"
                "*****************************************************************\n"
                "********* Long training unconverged !!! max_delta = %f\n"
                "*****************************************************************\n"
                "*****************************************************************\n",
                self.arr_max_delta_q_long[i_iteration],
            )
        self.model_calculation_time = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def make_policy_table_long(self) -> None:
        """
        Creates policy table

        Returns:
            None
        """
        self.ptb = np.nanargmax(self.qtb, -1)
        self.pdf = (
            pd.DataFrame(
                self.ptb,
                columns=self.env.arr_status_name[
                    : self.env.observation_space["hvac"].n
                ],
                index=list(np.round(self.arr_temp_ticks, 2)) + ["n/a"],
            )
            .rename_axis("indoor temp")
            .replace(self.env.dict_index_action)
        )

    def save_model_pickle(self) -> None:
        """
        Saves the trained model to pickle file

        Returns:
            None
        """
        file_name = (
            f"model_"
            f"set=[{self.env.setpoint_temp_low}-{self.env.setpoint_temp_high}]_"
            f"grid=[{self.env.temp_min}-{self.env.temp_max}]_"
            f"gamma={self.gamma}_"
            f"outdoor={self.env.temp_outdoor_nominal}"
            f".pkl"
        )
        model_dir_path = get_dir_path("models")
        file_path = os.path.join(model_dir_path, file_name)
        with open(file_path, "wb") as model_file:
            pickle.dump(
                {
                    "model_calculation_time": self.model_calculation_time,
                    "setpoint_temp_low": self.env.setpoint_temp_low,
                    "setpoint_temp_high": self.env.setpoint_temp_high,
                    "temp_outdoor_nominal": self.env.temp_outdoor_nominal,
                    "gamma": self.gamma,
                    "theta": self.theta,
                    "temp_min": self.env.temp_min,
                    "temp_max": self.env.temp_max,
                    "temp_step": self.temp_step,
                    "arr_max_delta_q_long": self.arr_max_delta_q_long,
                    "len_arr_max_delta_q_long": self.len_arr_max_delta_q_long,
                    "qtb": self.qtb,
                    "vtb": self.vtb,
                    "ptb": self.ptb,
                    "pdf": self.pdf,
                    "n_iteration_long": self.n_iteration_long,
                },
                model_file,
            )

    def train(self) -> None:
        """
        Sequence of methods for agent training

        Returns:
            None
        """
        self.dp_long()
        self.make_policy_table_long()
        self.save_model_pickle()
