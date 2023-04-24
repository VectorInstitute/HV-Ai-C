import datetime
from os import path
from pathlib import Path
import sys
import wandb
import yaml
import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from benchmark_scripts.train import FilterObservation, QLearningAgent, create_env

# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "num_tiles sweep",
    "metric": {"goal": "maximize", "name": "rollout/ep_rew_mean"},
    "parameters": {
        "out_tmp_degree_res": {"values": [1, 2, 3, 5, 10]},
        "in_tmp_degree_res": {"values": [1, 2, 3, 5, 10]},
        "energy_weight": {"values": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(
    sweep=sweep_configuration, entity="vector-institute-aieng", project="hnp"
)


def main():
    conf_path = path.join(path.dirname(__file__), "configs/tuning.yaml")
    with open(conf_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    env_config = config["env"]
    agent_config = config["agent"]

    experiment_name = f"Tuning_{env_config['name']}_{'HNP-' if agent_config.get('hnp') else ''}{agent_config['name']}_{env_config['reward_type']}_{datetime.datetime.now():%Y-%m-%d}"

    # Wandb configuration
    wandb_run = wandb.init(
        name=experiment_name,
        config=wandb.config,
        job_type="tuning",
        group=env_config["name"],
    )
    wandb.define_metric("step")
    wandb.define_metric("episode")
    wandb.define_metric("rollout/*", step_metric="episode")
    wandb.define_metric("time/*", step_metric="step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("reward/*", step_metric="step")

    env_config["energy_weight"] = wandb.config.energy_weight
    env = create_env(env_config)
    if env_config["obs_to_keep"] is not None:
        env = FilterObservation(env, env_config["obs_to_keep"])

    n_timesteps_episode = (
        env.simulator._eplus_one_epi_len / env.simulator._eplus_run_stepsize
    )
    total_timesteps = agent_config["num_episodes"] * n_timesteps_episode

    out_tmp_rng = env.ranges["Site Outdoor Air Drybulb Temperature(Environment)"]
    in_tmp_rng = env.ranges["Zone Air Temperature(SPACE1-1)"]

    out_tmp_tiles_n = (
        out_tmp_rng[1] - out_tmp_rng[0]
    ) // wandb.config.out_tmp_degree_res + 1
    in_tmp_tiles_n = (
        in_tmp_rng[1] - in_tmp_rng[0]
    ) // wandb.config.in_tmp_degree_res + 1

    agent_config["num_tiles"] = [int(out_tmp_tiles_n), 20, int(in_tmp_tiles_n), 20]

    model = QLearningAgent(
        env,
        {"agent": agent_config, "env": env_config},
        total_timesteps,
        np.array(env_config["mask"]),
        agent_config["hnp"],
    )
    model.train()


# Start sweep job.
wandb.agent(sweep_id, function=main)
