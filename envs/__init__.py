from gym.envs.registration import register

import sinergym
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from pathlib import Path

idf_path = Path(__file__).parents[1] / "envs/data/buildings/"

register(
    id='Eplus-5Zone-hot-discrete-train-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': str(idf_path) + '/Train/' '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-train-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

register(
    id='Eplus-5Zone-hot-discrete-test-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': str(idf_path) + '/5ZoneAutoDXVAV_Test.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-test-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})
