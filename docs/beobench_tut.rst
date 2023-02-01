=================
Beobench Tutorial
=================

This tutorial will walk you through the steps to training a RL agent using the HNP package inside a Beobench container. 

.. tip::
    Make sure you follow the installation instructions in Beobench `repository <https://github.com/rdnfn/beobench>`_ or `documentation <https://beobench.readthedocs.io/en/latest/>`_ to set up your BeoBench environment correctly.

1.  Navigate to ``/examples/beobench``, and you will see the following files:

  * ``beobench_config.yaml``: Agent and environment configuration file for HNP and Beobench. It has all the settings shown in the configuration page, but the structure follows the Beobench format. For more information on Beobench configuration requirements, check out this `link <https://beobench.readthedocs.io/en/latest/guides/configuration.html>`__.

  * ``train_beobench.py``: The RL agent training script.

  * ``beobench_env``: This directory contains Beobench customized environment. For more information on how to create your own environment in Beobench, check out this `link <https://beobench.readthedocs.io/en/latest/guides/add_env.html>`__.

    * ``Dockerfile``: Dockerfile for building the customized environment. In this example, we are still using Sinergym as the underlying building control environment, and included HNP package installation as well.

    * ``env_creator.py``: This file contains the environment creator function. This is the same as the function provided in the HNP package, but Beobench requires this file for customized environment.

2.  Run the following command to start training:
 
    .. code-block:: console

        beobench run -config beobench_config.yaml

    The building process could take a while if the docker images for your custom environment don't exist.

3.  Once the training completes, the rewards will be saved to the directory you specified in your config file followed by ``/yyyy_mon_dd/results_H_M_S/rewards.npy``. You can use ``examples/plot_results.py`` to plot the rewards by running:

    .. code-block:: console

        python plot_results path/to/rewards.npy 

    Alternatively, if you want to compare the rewards between different runs/agents, you can append the second rewards path to the above command to plot the difference.