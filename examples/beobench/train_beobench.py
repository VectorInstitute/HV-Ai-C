import numpy as np
from beobench.experiment.provider import config

from hnp.agents import QLearningAgent
from hnp.environment import ObservationWrapper, create_env


def main():
    obs_to_keep = np.array(config["env"]["config"]["obs_to_keep"])
    mask = np.array(config["env"]["config"]["mask"])

    env = create_env(config["env"]["config"])
    env = ObservationWrapper(env, obs_to_keep)

    agent = QLearningAgent(
        env, 
        config["agent"]["config"],
        mask,
        results_dir=config["general"]["local_dir"],
        use_beobench=True
    )
    agent.train()
    agent.save_results()
    env.close()

if __name__ == "__main__":
    main()