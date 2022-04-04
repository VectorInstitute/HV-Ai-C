"""
This file contains the Environment class.
"""
from typing import Tuple
import logging

import gym
import pandas as pd
import numpy as np

from src.util import read_config


class Environment(gym.Env):
    """
    This is the simulated environment class.
    """

    def __init__(
        self,
        temp_min: float = 10.0,
        temp_max: float = 20.0,
        setpoint_temp_low: float = 10.0,
        setpoint_temp_high: float = 20.0,
        temp_outdoor_nominal: float = 17.0,
        config_name: str = "environment_params.cfg",
    ) -> None:
        """
        Constructor for Environment class

        Args:
            temp_min (optional): min of temperature in the calculation. Defaults to 10.
            temp_max (optional): max of temperature in the calculation. Defaults to 20.
            setpoint_temp_low (optional): set point. Defaults to 10.
            setpoint_temp_high (optional): set point. Defaults to 20.
            temp_outdoor_nominal (optional): outdoor temperature as start, no use in
                real time running. Defaults to 17.
            config_name (optional): config file name. Defaults to environment_params.cfg

        Returns:
            None
        """
        super().__init__()

        self.parser = read_config(config_name)
        self.read_reward_params()
        self.read_state_transition_params()

        self.temp_outdoor_nominal = temp_outdoor_nominal

        self.temp_min = temp_min
        self.temp_max = temp_max
        self.setpoint_temp_low = setpoint_temp_low
        self.setpoint_temp_high = setpoint_temp_high

        self.define_one_time_device_start_reward()
        self.define_status_and_actions()
        self.define_dicts_status_index_devices()
        self.define_dict_status_to_power()
        self.define_observation_and_action_space()

        self.logger = logging.getLogger("root")

        self.temp = None
        self.hvac = None
        self.real_time = None
        self.temp_outdoor = None

    def read_reward_params(self) -> None:
        """
        Reward of Power:
        The action space contains 4 actions: idle, compressor on, freecool on,
        and heater on. The following parameters reflects the reward of each action
        based on its average power consumption from historical data.

        Reward of Device Start:
        Penalty coefficients for every time a device is turned on.

        Rewards of Temperature Offend:
        Penalty for going over the defined temperature range.

        Returns:
            None
        """

        # Reward of Power:
        self.power_idle = float(self.parser["reward"]["power_idle"])
        self.power_compressor_on = float(self.parser["reward"]["power_compressor_on"])
        self.power_freecool_on = float(self.parser["reward"]["power_freecool_on"])
        self.power_heater_on = float(self.parser["reward"]["power_heater_on"])

        # Reward of Device Start:
        self.multiplier_compressor_start = float(
            self.parser["reward"]["multiplier_compressor_start"]
        )
        self.multiplier_freecool_start = float(
            self.parser["reward"]["multiplier_freecool_start"]
        )
        self.multiplier_heater_start = float(
            self.parser["reward"]["multiplier_heater_start"]
        )

        # Rewards of Temperature Offend:
        self.reward_temp_offend = float(self.parser["reward"]["reward_temp_offend"])

    def read_state_transition_params(self) -> None:
        """
        The following parameters reflects the rate of temperature change for each type
        of transition. The coefficient and intercept are computed using linear regression
        where X is the difference between current outdoor temperature and current indoor
        temperature, and y is the difference between next minuture indoor temperature and
        current indoor temperature.

        The parameter suffix (all/pos_diff/neg_diff) indicates whether the temperature
        difference between indoor and outdoor is considered in the linear regression.
            'pos_diff': outdoor temperature > indoor temperature
            'neg_diff': outdoor temperature <= indoor temperature
            'all': disregard the difference

        Returns:
            None
        """

        # Compressor transition
        self.temp_step_compressor_intercept_all = float(
            self.parser["transition"]["temp_step_compressor_intercept_all"]
        )
        self.temp_step_compressor_coef_all = float(
            self.parser["transition"]["temp_step_compressor_coef_all"]
        )

        self.temp_step_compressor_intercept_neg_diff = float(
            self.parser["transition"]["temp_step_compressor_intercept_neg_diff"]
        )
        self.temp_step_compressor_coef_neg_diff = float(
            self.parser["transition"]["temp_step_compressor_coef_neg_diff"]
        )

        self.temp_step_compressor_intercept_pos_diff = float(
            self.parser["transition"]["temp_step_compressor_intercept_pos_diff"]
        )
        self.temp_step_compressor_coef_pos_diff = float(
            self.parser["transition"]["temp_step_compressor_coef_pos_diff"]
        )

        # Freecool transition
        self.temp_step_freecool_intercept_all = float(
            self.parser["transition"]["temp_step_freecool_intercept_all"]
        )
        self.temp_step_freecool_coef_all = float(
            self.parser["transition"]["temp_step_freecool_coef_all"]
        )

        self.temp_step_freecool_intercept_neg_diff = float(
            self.parser["transition"]["temp_step_freecool_intercept_neg_diff"]
        )
        self.temp_step_freecool_coef_neg_diff = float(
            self.parser["transition"]["temp_step_freecool_coef_neg_diff"]
        )

        self.temp_step_freecool_intercept_pos_diff = float(
            self.parser["transition"]["temp_step_freecool_intercept_pos_diff"]
        )
        self.temp_step_freecool_coef_pos_diff = float(
            self.parser["transition"]["temp_step_freecool_coef_pos_diff"]
        )

        # Heater transition
        self.temp_step_heater_intercept_all = float(
            self.parser["transition"]["temp_step_heater_intercept_all"]
        )
        self.temp_step_heater_coef_all = float(
            self.parser["transition"]["temp_step_heater_coef_all"]
        )

        self.temp_step_heater_intercept_neg_diff = float(
            self.parser["transition"]["temp_step_heater_intercept_neg_diff"]
        )
        self.temp_step_heater_coef_neg_diff = float(
            self.parser["transition"]["temp_step_heater_coef_neg_diff"]
        )

        self.temp_step_heater_intercept_pos_diff = float(
            self.parser["transition"]["temp_step_heater_intercept_pos_diff"]
        )
        self.temp_step_heater_coef_pos_diff = float(
            self.parser["transition"]["temp_step_heater_coef_pos_diff"]
        )

        # Idle transition when indoor temperature is above 18 degrees
        self.temp_step_idle_intercept_all_above_th = float(
            self.parser["transition"]["temp_step_idle_intercept_all_above_th"]
        )
        self.temp_step_idle_coef_all_above_th = float(
            self.parser["transition"]["temp_step_idle_coef_all_above_th"]
        )

        self.temp_step_idle_intercept_neg_diff_above_th = float(
            self.parser["transition"]["temp_step_idle_intercept_neg_diff_above_th"]
        )
        self.temp_step_idle_coef_neg_diff_above_th = float(
            self.parser["transition"]["temp_step_idle_coef_neg_diff_above_th"]
        )

        self.temp_step_idle_intercept_pos_diff_above_th = float(
            self.parser["transition"]["temp_step_idle_intercept_pos_diff_above_th"]
        )
        self.temp_step_idle_coef_pos_diff_above_th = float(
            self.parser["transition"]["temp_step_idle_coef_pos_diff_above_th"]
        )

        # Idle transition when indoor temperature is less than or equal to 18 degrees
        self.temp_step_idle_intercept_all_below_th = float(
            self.parser["transition"]["temp_step_idle_intercept_all_below_th"]
        )
        self.temp_step_idle_coef_all_below_th = float(
            self.parser["transition"]["temp_step_idle_coef_all_below_th"]
        )

        self.temp_step_idle_intercept_neg_diff_below_th = float(
            self.parser["transition"]["temp_step_idle_intercept_neg_diff_below_th"]
        )
        self.temp_step_idle_coef_neg_diff_below_th = float(
            self.parser["transition"]["temp_step_idle_coef_neg_diff_below_th"]
        )

        self.temp_step_idle_intercept_pos_diff_below_th = float(
            self.parser["transition"]["temp_step_idle_intercept_pos_diff_below_th"]
        )
        self.temp_step_idle_coef_pos_diff_below_th = float(
            self.parser["transition"]["temp_step_idle_coef_pos_diff_below_th"]
        )

    def define_one_time_device_start_reward(self) -> None:
        """
        Computes the device start penalty

        Returns:
            None
        """
        self.reward_power_compressor_start = -self.multiplier_compressor_start * (
            self.power_compressor_on - self.power_idle
        )
        self.reward_power_freecool_start = -self.multiplier_freecool_start * (
            self.power_freecool_on - self.power_idle
        )
        self.reward_power_heater_start = -self.multiplier_heater_start * (
            self.power_heater_on - self.power_idle
        )

    def define_status_and_actions(self) -> None:
        """
        Define device status (state) and actions

        Returns:
            None
        """
        # status
        self.arr_status_name = pd.Series(
            [
                "status_idle",
                "status_compressor_on",
                "status_freecool_on",
                "status_heater_on",
            ]
        )
        self.dict_index_status = dict(self.arr_status_name)
        (
            self.status_idle,
            self.status_compressor_on,
            self.status_freecool_on,
            self.status_heater_on,
        ) = self.arr_status_name.index
        self.status_size = len(self.arr_status_name)

        # actions
        self.arr_action_name = pd.Series(
            [
                "action_idle",
                "action_compressor_on",
                "action_freecool_on",
                "action_heator_on",
            ]
        )
        self.dict_index_action = dict(self.arr_action_name)
        (
            self.action_idle,
            self.action_compressor_on,
            self.action_freecool_on,
            self.action_heator_on,
        ) = self.arr_action_name.index
        self.action_size = len(self.arr_action_name)

    def define_dicts_status_index_devices(self) -> None:
        """
        Encode device status

        Returns:
            None
        """
        self.dic_status_to_devices = {
            self.status_compressor_on: (1, 0, 0),
            self.status_freecool_on: (0, 1, 0),
            self.status_heater_on: (0, 0, 1),
            self.status_idle: (0, 0, 0),
        }
        self.dic_devices_to_status = {
            tup_devices: status
            for status, tup_devices in self.dic_status_to_devices.items()
        }

    def define_dict_status_to_power(self) -> None:
        """
        Create mapping between device status and action

        Returns:
            None
        """
        self.dic_status_to_power = {
            self.status_compressor_on: self.power_compressor_on,
            self.status_freecool_on: self.power_freecool_on,
            self.status_heater_on: self.power_heater_on,
            self.status_idle: self.power_idle,
        }

    def define_observation_and_action_space(self) -> None:
        """
        Create gym environment

        Returns:
            None
        """
        self.observation_space = gym.spaces.Dict(
            {
                "temp": gym.spaces.Box(
                    low=np.float64(self.temp_min),
                    high=np.float64(self.temp_max),
                    shape=(1,),
                    dtype=np.float64,
                ),
                "hvac": gym.spaces.Discrete(self.status_size),
            }
        )
        self.action_space = gym.spaces.Discrete(self.action_size)

    def calculate_next_state(self, temp: float, action: int) -> Tuple[float, int]:
        """
        Find the next state based on current temperature and action

        Args:
            temp: Current temperature
            action: Current action

        Returns:
            Next state temperature and next state HVAC status
        """
        if action == self.action_compressor_on:
            next_hvac = self.status_compressor_on
            intercept = self.temp_step_compressor_intercept_all
            coef = self.temp_step_compressor_coef_all

        elif action == self.action_freecool_on:
            next_hvac = self.status_freecool_on
            if self.temp_outdoor <= temp:
                intercept = self.temp_step_freecool_intercept_neg_diff
                coef = self.temp_step_freecool_coef_neg_diff
            else:
                intercept = self.temp_step_freecool_intercept_pos_diff
                coef = self.temp_step_freecool_coef_pos_diff

        elif action == self.action_heator_on:
            next_hvac = self.status_heater_on
            intercept = self.temp_step_heater_intercept_neg_diff
            coef = self.temp_step_heater_coef_neg_diff

        else:  # aciton == idle
            next_hvac = self.status_idle
            if temp > 18.0:
                if self.temp_outdoor <= temp:
                    intercept = self.temp_step_idle_intercept_neg_diff_above_th
                    coef = self.temp_step_idle_coef_neg_diff_above_th
                else:
                    intercept = self.temp_step_idle_intercept_pos_diff_above_th
                    coef = self.temp_step_idle_coef_pos_diff_above_th
            else:
                if self.temp_outdoor <= temp:
                    intercept = self.temp_step_idle_intercept_neg_diff_below_th
                    coef = self.temp_step_idle_coef_neg_diff_below_th
                else:
                    intercept = self.temp_step_idle_intercept_pos_diff_below_th
                    coef = self.temp_step_idle_coef_pos_diff_below_th

        temp_increase = intercept + coef * (self.temp_outdoor - temp)
        next_temp = float(np.clip(temp + temp_increase, self.temp_min, self.temp_max))

        return next_temp, next_hvac

    def calculate_power(self, hvac: int = None) -> float:
        """
        Calculate HVAC power consumption

        Args:
            hvac (optional): HVAC status Defaults to None.

        Returns:
            Power consumed
        """
        if hvac is None:
            hvac = self.hvac
        power = self.dic_status_to_power[hvac]
        return power

    def calculate_reward(self, temp: float, hvac: int, action: int) -> float:
        """
        Compute current reward

        Args:
            temp: Current temperature
            hvac: Current HVAC status
            action: Current action

        Returns:
            Current reward
        """
        # reward 1: normal power consumption
        reward_power = -self.calculate_power(hvac)

        # reward 2: one time compressor or free cool start
        if hvac != self.status_compressor_on and action == self.action_compressor_on:
            reward_device_start = self.reward_power_compressor_start

        elif hvac != self.status_freecool_on and action == self.action_freecool_on:
            reward_device_start = self.reward_power_freecool_start

        elif hvac != self.status_heater_on and action == self.action_heator_on:
            reward_device_start = self.reward_power_heater_start

        else:
            reward_device_start = 0

        # reward 3: temp violation
        if temp <= self.setpoint_temp_low:
            reward_temp_offend = self.reward_temp_offend * (
                self.setpoint_temp_low - temp + 1.0
            )

        elif temp >= self.setpoint_temp_high:
            reward_temp_offend = self.reward_temp_offend * (
                temp - self.setpoint_temp_high + 1.0
            )

        else:
            reward_temp_offend = 0

        return reward_power + reward_device_start + reward_temp_offend

    def calculate_next_state_and_reward(
        self, action: int, temp: float = None, hvac: int = None
    ) -> Tuple[Tuple[float, int], float]:
        """
        Sequence of methods for state transiiton

        Args:
            action: Current action
            temp (optional): Current temperature. Defaults to None.
            hvac (optional): Current HVAC status. Defaults to None.

        Returns:
            Next state and current reward
        """
        if temp is None:
            temp = self.temp
        if hvac is None:
            hvac = self.hvac

        # calculate next state
        next_temp, next_hvac = self.calculate_next_state(temp, action)

        # calculate reward
        reward = self.calculate_reward(temp, hvac, action)

        return (next_temp, next_hvac), reward
