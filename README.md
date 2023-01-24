# Hyperspace Neighbour Penetration for Energy Efficient HVAC Control

Data center temperature control is a critical process for maintaining high quality of service. While maintaining temperatures in appropriate temperature bands is paramount, it is also important to consider that every heating or cooling strategy has associated environmental and economic costs. For example, a cooling strategy that makes excessive or unnecessary use of air conditioning will consume more energy than a strategy that better leverages free cooling. As long as operational constraints are satisfied, opportunities to discover temperature control solutions that minimize energy utilization are highly valuable.

This repository provides a reference solution for training a reinforcement learning (RL) agent to perform temperature control in a room. The algorithm leverages an innovation in state space design called **Hyperspace Neighbour Penetration** (HNP).

A key assumption in HNP is that continuous state spaces with very incremental, locally linear transitions can be effectively discretized into relatively coarse ranges or bands of values as *tiles*. When this type of coarse discretization is applied to a slowly-changing variable (e.g. temperature), it can lead to situations where an action (e.g. changeing the setpoints) results in such a small change to the subsequent state observation that no discrete state transition has actually occurred. A naïve solution could be to increase the granularity of the state space, i.e. to consider much smaller changes in temperature as distinct elements of the state space, but it is computationally expensive or impossible to establish an extremely granular grid system. Alternatively, HNP computes multiple values from tile *boundaries* and then aggregates them using a weighted norm. This enables state-action pairs to result in *steps towards* other states and their corresponding values. 

HNP is fully described in its [foundational paper](https://arxiv.org/pdf/2106.05497.pdf).

# Dependencies

* Python ≥ 3.9
* [Sinergym](https://github.com/ugr-sail/sinergym)
* [NumPy](https://github.com/numpy/numpy)

# Supported Agents

The HNP package provides the following agents:

* Random Action Agent: An agent that takes a random action 
* Fixed Action Agent: An agent that always take a pre-defined action
* HNP-enabled Q-Learning Agent: A Q-learning agent with built-in HNP that allows different types of observation variables