=================
Sinergym Tutorial
=================

This tutorial will walk you through the steps to a RL agent using the HNP package inside the Sinergym environment. 

.. tip::
    Make sure you follow the installation instructions in Sinergym `repository <https://github.com/ugr-sail/sinergym>`_ or `documentation <https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html>`_ to set up your Sinergym environment correctly.

1.  Navigate to ``/examples/sinergym``, and you will see the ``sinergym_config.yaml`` and ``train_sinergym.py``. These are the only files you need to train an agent in Sinergym.
2.  Change the configuration in ``sinergym_config.yaml`` according to your needs.
3.  Run the following command to start training:
 
    .. code-block:: console

        python train_sinergym.py sinergym_config.yaml

4.  Once the training completes, the rewards will be saved to ``training_results/yyyy_mon_dd/results_H_M_S/rewards.npy``. You can use ``examples/plot_results.py`` to plot the rewards by running:

    .. code-block:: console

        python plot_results path/to/rewards.npy 

    Alternatively, if you want to compare the rewards between different runs/agents, you can append the second rewards path to the above command to plot the difference.