import sys
import json

import matplotlib.pyplot as plt
import numpy as np


def main(filenames: list[str]) -> None:
    if len(filenames) > 2:
        plot_diff(filenames[1:])
    else:
        if ".npz" in filenames[1]:
            results = np.load(filenames[1])
            agent_name = filenames[1].split("/")[-1].split("_")[0]
            for f in results.files:
                if "rewards" in f:
                    plot_rewards(agent_name, results[f])
                    break
        elif ".npy" in filenames[1]:
            filename = filenames[1].split("/")[-1]
            agent_name = filename.split("_")[0]
            rewards = np.load(filename)
            plot_rewards(agent_name, rewards)
        elif ".json" in filenames[1]:
            plot_dqn(filenames[1])
        else:
            print("Unrecognized file extension")


def plot_rewards(agent_name: str, rewards: np.ndarray) -> None:
    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(7)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Episodic Rewards")
    plt.title(f"{agent_name} Training Rewards")
    # plt.show()
    plt.savefig(f"{agent_name}_rewards.png")
    plt.close()


def plot_diff(filenames: list[str]) -> None:
    filename_0 = filenames[0].split("/")[-1]
    filename_1 = filenames[1].split("/")[-1]

    agent_1 = filename_0.split("_")[0]
    agent_2 = filename_1.split("_")[0]

    rewards_1 = np.load(filename_0)
    rewards_2 = np.load(filename_1)
    rewards_diff = rewards_1 - rewards_2

    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(7)

    plt.plot(rewards_diff)
    plt.xlabel("Episodes")
    plt.ylabel("Episodic Rewards Diff")
    plt.title(f"{agent_1} vs {agent_2} Training Rewards Diff")
    plt.rcParams["figure.figsize"] = (35,10)
    # plt.show()
    plt.savefig(f"{agent_1}_v_{agent_2}_rewards.png")
    plt.close()


def plot_dqn(filename: str) -> None:
    with open(filename) as results_json:
        results = json.load(results_json)
    rewards = []
    for iter in results:
        rewards.extend(iter["sampler_results"]["hist_stats"]["episode_reward"])
    episode_index = np.arange(0, len(rewards), 1, dtype=int)
    plt.plot(episode_index, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Episodic Rewards")
    plt.title("DQN Training Rewards")
    # plt.show()
    plt.savefig("dqn_rewards.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing data file")
    else:
        main(sys.argv)