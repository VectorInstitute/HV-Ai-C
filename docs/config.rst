=============
Configuration
=============

HNP uses dictionary to parse agent and environment configuration. You can use any file format you are comfortable with (``YAML``, ``JSON``, etc.). Below is a sample configuration in ``YAML`` format. The comments above each setting describe its functionality.

.. code-block:: YAML

    agent:
        # The number of episodes for training
        num_episodes: 1460
        # Maximum number of time steps in each episode, each time step lasts 15 minutes in Sinergym
        horizon: 24
        # Discount factor
        gamma: 0.99
        # Number of tiles for tile coding. Use a integer for the same number of tiles across all continuous variables. 
        # Use a list for distinct number of tiles for different continuous variables, each number in the list must match the order in obs_to_keep
        num_tiles: 20
        # Initial value for Epsilon
        initial_epsilon: 1
        # Annealing rate for Epsilon
        epsilon_annealing: 0.999
        # Initial learning rate
        learning_rate: 0.1
        # Annealing rate for learning rate
        learning_rate_annealing: 0.999
        # The action index for training, this is only used for fixed action agent
        action_index: 9
            
    env:
        # Sinergym environment name
        name: Eplus-5Zone-hot-discrete-v1
        # Whether to normalise observations
        normalize: True
        # The observation variables to use for training. Use a empty list if you want to use all observation variables
        obs_to_keep: [1, 2, 8, 10]
        # The type of each observation variable: 
        #   0 - slowly-changing continuous variable
        #   1 - fast-changing continuous variable
        #   2 - discrete variable
        mask: [0, 0, 0, 0]

