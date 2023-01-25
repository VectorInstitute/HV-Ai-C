=================
Sinergym Tutorial
=================

This tutorial will walk you through the steps to training a RL agent using the HNP package inside the Sinergym environment. 

.. tip::
    Make sure you follow the installation instructions in Sinergym `repository <https://github.com/ugr-sail/sinergym>`_ or `documentation <https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html>`_ to set up your Sinergym environment correctly.

1.  Navigate to ``/examples/sinergym``, and you will see the following files:
    
    * ``sinergym_config.yaml``:  Agent and environment configuration file for HNP.
    
    * ``train_sinergym.py`` : The RL agent training script.

2.  Run the following command to start training:
 
    .. code-block:: console

        python train_sinergym.py sinergym_config.yaml

3.  Once the training completes, the rewards will be saved to ``training_results/yyyy_mon_dd/results_H_M_S/rewards.npy``. You can use ``examples/plot_results.py`` to plot the rewards by running:

    .. code-block:: console

        python plot_results path/to/rewards.npy 

    Alternatively, if you want to compare the rewards between different runs/agents, you can append the second rewards path to the above command to plot the difference.