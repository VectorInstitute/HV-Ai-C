"""
This file contains the main training loop
"""
import sys
import datetime

import numpy as np

from src.util import init_logger, create_dirs, read_config
from src.agent import Agent
from src.environment import Environment


def main(training_config_name: str = "training_params.cfg") -> None:
    """
    Main training loop

    Args:
        training_config_name (optional): Defaults to "training_params.cfg".

    Returns:
        None
    """
    create_dirs()
    logger = init_logger("root")

    parser = read_config(training_config_name)

    outdoor_temp_lo = float(parser["training_env"]["outdoor_temp_low"])
    outdoor_temp_hi = float(parser["training_env"]["outdoor_temp_high"])
    outdoor_temp_step = float(parser["training_env"]["outdoor_temp_step"])

    for temp_outdoor_nominal in np.arange(
        outdoor_temp_lo, outdoor_temp_hi, outdoor_temp_step
    ):
        logger.info(
            "Training model for outdoor temperature between %f and %f",
            temp_outdoor_nominal,
            temp_outdoor_nominal + 5,
        )
        logger.info(
            "temp_outdoor_nominal = %f; start dp_long at %s",
            temp_outdoor_nominal,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        env = Environment(
            temp_min=float(parser["training_env"]["temp_min"]),
            temp_max=float(parser["training_env"]["temp_max"]),
            setpoint_temp_low=float(parser["training_env"]["setpoint_temp_low"]),
            setpoint_temp_high=float(parser["training_env"]["setpoint_temp_high"]),
            temp_outdoor_nominal=temp_outdoor_nominal,
        )
        agent = Agent(
            env,
            temp_step=float(parser["training_agent"]["temp_step"]),
            gamma=float(parser["training_agent"]["gamma"]),
            theta=float(parser["training_agent"]["theta"]),
            n_iteration_long=int(1e6),
        )
        agent.train()
        logger.info(
            "temp_outdoor_nominal = %f; finish dp_long at %s",
            temp_outdoor_nominal,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
