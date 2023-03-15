import argparse
import datetime
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
    parser.add_argument("--env-config", type=str,
                        default="configs/environments.yaml", help="Environment config file")
    parser.add_argument("--wandb", action="store_true",
                        help="Log using WandB")

    args = parser.parse_args()

    if args.env_config.startswith("/"):
        env_config_path = args.env_config
    else:
        env_config_path = path.join(path.dirname(__file__), args.env_config)

    with open(env_config_path, "r") as env_config_file:
        env_config = yaml.safe_load(env_config_file)

    ALGOS = {"QLearning": QLearningAgent, "DQN": DQN}

    model_name = args.model_path.split("/")[-1]
    model_config = model_name.split("_")
    env_name = model_config[0].replace("train", "test")
    algo_name = model_config[1]

    env_config = env_config[env_name]

    env = create_env(env_config)
    if env_config['obs_to_keep'] is not None:
        env = FilterObservation(env, env_config['obs_to_keep'])

    agent = "QLearning" if "HNP" in algo_name else algo_name
    model = ALGOS[agent].load(args.model_path, env)

    experiment_name = f"{env_config['name']}_{algo_name}_{env_config['reward_type']}_{args.num_eval}_{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"

    if args.wandb:
        wandb_run = wandb.init(name=experiment_name, project="hnp", entity="vector-institute-aieng",
                               config=model.__dict__, sync_tensorboard=True)

    returns = []
    for i in trange(args.num_eval):
        total_reward = 0
        obs = env.reset()
        while True:
            action = model.predict(obs)
            if not isinstance(action, np.int64):
                action = int(action[0])
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                returns.append(total_reward)
                if wandb.run:
                    wandb.log({"rollout/ep_rew_mean": np.mean(np.array(returns)),
                               "rollout/ep_return": total_reward})
                break
    returns = np.array(returns)
    env.close()
    if wandb.run:
        wandb.finish()
    print(f"Environment: {env_name}\nAgent: {algo_name}\nNum_Eval_Episodes: {args.num_eval}\nMean reward: {np.mean(returns)}\nStd reward: {np.std(returns)}")
