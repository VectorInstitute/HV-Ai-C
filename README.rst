.. start-in-sphinx-home-docs

==============================================================================================
Reinforcement Learning with Hyperspace Neighbour Penetration for Energy Efficient HVAC Control
==============================================================================================

.. image:: https://img.shields.io/pypi/v/hnp.svg
        :target: https://pypi.python.org/pypi/hnp

.. image:: https://readthedocs.org/projects/hnp/badge/?version=latest
        :target: https://hnp.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/hnp.svg
        :target: https://opensource.org/licenses/BSD-3-Clause
        :alt: License

Data center temperature control is a critical process for maintaining high quality of service. While maintaining temperatures in appropriate temperature bands is paramount, it is also important to consider that every heating or cooling strategy has associated environmental and economic costs. For example, a cooling strategy that makes excessive or unnecessary use of air conditioning will consume more energy than a strategy that better leverages free cooling. As long as operational constraints are satisfied, opportunities to discover temperature control solutions that minimize energy utilization are highly valuable. Co-developed by the **Vector Institute** and **TELUS**, this repository provides a solution for training a reinforcement learning (RL) agent to perform temperature control in a room. Click `here <https://hnp.readthedocs.io>`__ for detailed documentation.

Features
========
- **Innovative discretization method**: The provided solution leverages an innovation in state space design proposed by TELUS called **Hyperspace Neighbour Penetration** (HNP), which allows slowly-changing continuous variables to be approximated as discrete variables.
- **Support for building control environments**: The HNP package has built-in support for Sinergym - a building control simulation environment, as well as Beobench - a toolkit providing unified access to building control environments.
- **Ready-to-use RL agents**: A HNP-enabled Q-Learning agent and two baseline agents are provided in the package.

.. end-in-sphinx-home-docs

.. start-in-sphinx-getting-started-overview

HNP
===

A key assumption in HNP is that continuous state spaces with very incremental, locally linear transitions can be effectively discretized into relatively coarse ranges or bands of values as *tiles*. When this type of coarse discretization is applied to a slowly-changing variable (e.g. temperature), it can lead to situations where an action (e.g. changing the setpoints) results in such a small change to the subsequent state observation that no discrete state transition has actually occurred. A naïve solution could be to increase the granularity of the state space, i.e. to consider much smaller changes in temperature as distinct elements of the state space, but it is computationally expensive or impossible to establish an extremely granular grid system. Alternatively, HNP computes multiple values from tile *boundaries* and then aggregates them using a weighted norm. This enables state-action pairs to result in *steps towards* other states and their corresponding values. 

HNP is fully described in its `foundational paper <https://arxiv.org/pdf/2106.05497.pdf>`_.

Supported Environments
======================

Sinergym
--------

`Sinergym <https://github.com/ugr-sail/sinergym>`_ is a building control environment that follows OpenAI Gym interface and uses EnergyPlus simulator. To use Sinergym, see detailed instruction on how to install `here <https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html>`__.

Beobench
--------

`Beobench <https://github.com/rdnfn/beobench>`_ is a toolkit providing unified access to building control environments for RL (Sinergym also supported). It uses docker to manage all environment dependencies in the background. See detailed instruction on how to use Beobench `here <https://beobench.readthedocs.io/en/latest/>`__.

Supported Agents
================

The HNP package provides the following agents:

- Random Action Agent: An agent that takes a random action 
- Fixed Action Agent: An agent that always take a pre-defined action
- HNP-enabled Q-Learning Agent: A Q-learning agent with built-in HNP that allows different types of observation variables

.. end-in-sphinx-getting-started-overview


Quickstart
============

Requirements
------------
- Python ≥ 3.9
- Sinergym == 2.2.0

.. start-in-sphinx-getting-started-quickstart

Installation
------------

To install ``hnp`` from `PyPI <https://pypi.org/project/hnp/>`_:

.. code-block:: console

    pip install hnp

Example Usage
-------------

This is a minimalist example of using the HNP Q-Learning agent in Sinergym

.. code-block:: python

    import numpy as np

    from hnp.agents import QLearningAgent
    from hnp.environment import ObservationWrapper, create_env

    config = {
        "agent": {
            "num_episodes": 100,
            "horizon": 24,
            "gamma": 0.99,
            "num_tiles": 20,
            "initial_epsilon": 1,
            "epsilon_annealing": 0.999,
            "learning_rate": 0.1,
            "learning_rate_annealing": 0.999
        },
        "env": {
            "name": "Eplus-5Zone-hot-discrete-v1",
            "normalize": True,
            "obs_to_keep": [4, 5, 13],
            "mask": [0, 0, 0]
        }
    }

    obs_to_keep = np.array(config["env"]["obs_to_keep"])
    mask = np.array(config["env"]["mask"])

    env = create_env(config["env"])
    env = ObservationWrapper(env, obs_to_keep)

    agent = QLearningAgent(
        env, 
        config["agent"]["params"],
        mask,
    )
    agent.train()
    agent.save_results()
    env.close()

.. end-in-sphinx-getting-started-quickstart

Documentation
=============
Detailed package documentation: `HNP Docs <https://hnp.readthedocs.io>`_

License
=======
The code in this repository is published under 3-Clause BSD license (see ``LICENSE`` file).
