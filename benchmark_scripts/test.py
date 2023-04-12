import argparse
import datetime
import pathlib
import wandb
from os import path
from pathlib import Path
import sys
import yaml
import numpy as np
from stable_baselines3 import DQN
from tqdm import trange

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from benchmark_scripts.train import FilterObservation, create_env, QLearningAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RL-HVAC-Control-Benchmark-Test")

    parser.add_argument("--model-path", type=str,
                        required=True, help="Trained model path")
    parser.add_argument("--num-eval", type=int,
                        default=50, help="Number of evaluation episodes")
    parser.add_argument("--wandb", action="store_true",
                        help="Log using WandB")

    args = parser.parse_args()
    *_, load_path = pathlib.Path(args.model_path).iterdir()
    load_path = str(load_path)
    with open(load_path + "/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    ALGOS = {"QLearning": QLearningAgent, "DQN": DQN}

    env_name = config["env_config"]["name"].replace("train", "test")
    algo_name = config["agent_config"]["name"]

    env_config = config["env_config"]

    env = create_env(env_config)
    if env_config['obs_to_keep'] is not None:
        env = FilterObservation(env, env_config['obs_to_keep'])

    model = ALGOS[algo_name].load(load_path + "/policy", env)

    experiment_name = f"{env_name}_{'HNP-' if config['agent_config'].get('hnp') else ''}{algo_name}_{env_config['reward_type']}_{datetime.datetime.now():%Y-%m-%d}"

    if args.wandb:
        wandb_run = wandb.init(name=experiment_name, project="hnp", entity="vector-institute-aieng",
                               config=model.__dict__, job_type="eval", group=env_name)
        wandb.define_metric("step")
        wandb.define_metric("episode")
        wandb.define_metric("rollout/*", step_metric="episode")
        wandb.define_metric("time/*", step_metric="step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("action_idx", step_metric="step")
        # wandb.define_metric("rewards/*", step_metric="step")
        wandb.Table.MAX_ROWS = 200000

    n_timesteps_episode = int(env.simulator._eplus_one_epi_len /
                              env.simulator._eplus_run_stepsize)
    returns = []
    total_power = np.zeros((args.num_eval, n_timesteps_episode))
    out_temp = np.zeros((args.num_eval, n_timesteps_episode))
    abs_comfort = np.zeros((args.num_eval, n_timesteps_episode))
    comfort_penalty = np.zeros((args.num_eval, n_timesteps_episode))
    total_power_no_units = np.zeros((args.num_eval, n_timesteps_episode))
    temp = np.zeros((args.num_eval, n_timesteps_episode))
    total_action_idx = [[] for _ in range(n_timesteps_episode)]
    total_timesteps = 0
    for i in trange(args.num_eval):
        total_reward = 0
        obs = env.reset()
        timestep = 0
        while True:
            action = model.predict(obs)
            if not isinstance(action, np.int64):
                action = int(action[0])
            total_action_idx[timestep].append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            total_power[i, timestep] = info["total_power"]
            out_temp[i, timestep] = info["out_temperature"]
            temp[i, timestep] = info["temperatures"][0]
            comfort_penalty[i, timestep] = info["comfort_penalty"]
            abs_comfort[i, timestep] = info["abs_comfort"]
            total_power_no_units[i,
                                 timestep] = info["total_power_no_units"]
            if wandb.run:
                log_dict = {"action_idx": action,
                            "step": total_timesteps}
                wandb.log(log_dict)
            # if wandb.run:
            #     log_dict = {"rewards/total_power": total_power[i, timestep], "rewards/out_temp": out_temp[i, timestep], "rewards/temp": temp[i, timestep],
            #                 "rewards/abs_comfort": abs_comfort[i, timestep], "rewards/comfort_penalty": comfort_penalty[i, timestep], "rewards/total_power_no_units": total_power_no_units[i, timestep], "step": timestep}
            #     wandb.log(log_dict)
            timestep += 1
            total_timesteps += 1
            if done:
                returns.append(total_reward)
                if wandb.run:
                    log_dict = {"rollout/ep_rew_mean": np.mean(np.array(returns)),
                                "rollout/ep_return": total_reward, "episode": i}
                    wandb.log(log_dict)
                break
    if wandb.run:
        avg_total_power = np.mean(total_power, axis=0)
        avg_comfort_penalty = np.mean(comfort_penalty, axis=0)
        avg_abs_comfort = np.mean(abs_comfort, axis=0)
        avg_out_tmp = np.mean(out_temp, axis=0)
        avg_total_power_no_units = np.mean(total_power_no_units, axis=0)
        avg_tmp = np.mean(temp, axis=0)
        for i in range(n_timesteps_episode):
            wandb.log({"reward/total_power": avg_total_power[i], "reward/comfort_penalty": avg_comfort_penalty[i], "reward/abs_comfort": avg_abs_comfort[i], "reward/out_tmp": avg_out_tmp[i],
                       "reward/total_power_no_units": avg_total_power_no_units[i], "reward/tmp": avg_tmp[i], "train/ep_action_idx": wandb.Histogram(total_action_idx[i]), "step": i})
    returns = np.array(returns)
    env.close()
    if wandb.run:
        wandb.finish()
    print(f"Environment: {env_name}\nAgent: {algo_name}\nNum_Eval_Episodes: {args.num_eval}\nMean reward: {np.mean(returns)}\nStd reward: {np.std(returns)}")
