# Model-Based Reinforcement Learning for Datacentre HVAC Control

This repository introduces a reference solution for training a **Model-Based Reinforcement Learning** (MBRL) agent to control the HVAC system in a small-room datacentre such that **energy utilization is minimized**. When this agent was piloted in a small site, cooling savings of 18% (excluding IT load) and heating savings of 10% (excluding IT load) were realized, for a combined estimated annual electricity savings of 11.5%.  Therefore, potentially this agent can be implemented to reduce electricity costs as well as harmful greenhouse gas emissions that contribute to climate change.

# Repository Organization

All solution files are located in the `src` directory, which are organized as follows:

* `agent.py`: Reinforcement learning agent using the Hyperspace Neighbor Penetration algorithm.
* `environment.py`: OpenAI Gym environment simulating a room with HVAC controls. Parameterized by a file `configs/environment_params.cfg`, which must be populated according to instructions below, under **State Transition Parameters**.
* `util.py`: Utility functions supporting training.
* `main.py`: Main training loop for the reinforcement learning agent.

# Dependencies

* Python ≥ 3.6
* [OpenAI Gym](https://github.com/openai/gym)
* [NumPy](https://github.com/numpy/numpy)
* [Pandas](https://github.com/pandas-dev/pandas)

# Solution Overview

Data centre temperature control is a critical process for maintaining high quality of service. While maintaining temperatures in appropriate temperature bands is paramount, it is also important to consider that every heating or cooling strategy has associated environmental and economic costs. For example, a cooling strategy that makes excessive or unnecessary use of air conditioning will consume more energy than a strategy that better leverages free cooling. Free cooling is the process of bypassing mechanical direct-expansion cooling by exchanging heat with the lower temperature outdoor air, making it a more cost- and energy-efficient approach to cooling. As long as operational constraints are satisfied, opportunities to discover temperature control solutions that minimize energy utilization are highly valuable.

This repository provides a reference solution for training a MBRL agent to perform temperature control in a small-room data centre. The algorithm leverages an innovation in state space design called **Hyperspace Neighbour Penetration** (HNP), which allows a conventional dynamic programming (DP) based Q-learning algorithm to quickly learn effective, energy-efficient policies for HVAC system control.

A key assumption in HNP is that continuous state spaces with very incremental, locally linear transitions can be effectively discretized into relatively coarse ranges or bands of values as *tiles*. When this type of coarse discretization is applied in a conventional Q-learning framework, it can lead to situations where an action (e.g. turning on a compressor for cooling) results in such a small change to the subsequent state observation that no discrete state transition has actually occurred. A naïve solution could be to increase the granularity of the state space, i.e. to consider much smaller changes in temperature as distinct elements of the state space, but this makes DP-based Q-learning intractable. Alternatively, HNP computes multiple values from tile *boundaries* and then aggregates them using a weighted norm. This enables state-action pairs to result in *steps towards* other states and their corresponding values. This sharply reduces the effective search space size for DP and makes the problem tractable to compute.

HNP is fully described in its [foundational paper](https://arxiv.org/pdf/2106.05497.pdf), and the Q-learning algorithm is discussed in greater detail below.

# Reinforcement Learning Algorithm Breakdown

## State-Action Space

The available state variables and their corresponding values for the environment are as follows:

### Temperature
This state variable tracks the temperature of the environment between the set values of **[low, high]** with a given step size.

### HVAC
This state variable tracks the status of the HVAC system with the possible discrete values of:\
\
**["status_idle", "status_compressor_on", "status_freecool_on", "status_heater_on"]**

### Actions
The possible actions from a given state are:\
\
**["action_idle", "action_compressor_on", "action_freecool_on", "action_heater_on"]**\
\
It should be noted, however, that not all actions are possible at every state; for instance, if we are at or near maximum allowable temperature, we cannot select "action_heater_on". Such illegal state-action pairs are not allowed during training.

### Rewards
There are three different types of negative reward associated with the temperature control environment:\
\
**1. Normal Power Consumption:** This reward penalizes the agent for consuming power\
**2. Device Start Reward:** A reward associated with turning on the compressor or the free cooling device. Note that turning the devices on requires more power than selecting the corresponding actions and this is a one-time reward.\
**3. Temperature Violation:** Penalizing going to an unallowable state such as a temperature higher than the set maximum.

At every state transition, the corresponding reward is a sum of the three types of reward stated.

## State Transition Parameters
There are five possible transitions corresponding to the four possible actions:  
  
**["compressor", "freecool", "heater", "idle_above_th", "idle_below_th"]**
* **"idle_above_th"** and **"idle_below_th"** represent two idle scenarios, where **above_th** indicates that the indoor temperature is above the threshold and  **below_th** indicates that the indoor temperature is at or below the threshold.  
  
Linear regression is used to compute the rate of temperature change (per minute per degree) for each type of transition. The difference between the current outdoor temperature and current indoor temperature is used as the explanatory variable, and the difference between the next minute indoor temperature and the current indoor temperature is used as the dependent variable. The gradient (**coef**) and intercept (**intercept**) of the fitted line are saved for later use.
  
Three different lines are fitted for each transition:  
  
**["pos_diff", "neg_diff", "all"]**  
  
* Where **""pos_diff"** is applied when the outdoor temperature is greater than the indoor temperature, **"neg_diff"** is applied when the outdoor temperature is the same as or less than the indoor temperature, and **"all"** ignores the outdoor indoor temperature difference.  
  
The value of these parameters should be saved in `environment_params.cfg`.

**Note:** code to compute the parameters required for `environment_params.cfg` using historical HVAC data is _not included_ in the initial release of this reference solution, since the current version has not been adapted to generalize outside of the TELUS environment. Replication of the approach is straightforward using the instructions just above, and reference code to do this will be provided in a subsequent release. 

# Hyperspace Neighbour Penetration Dynamic Programming

Below is the series of steps the Dynamic Programming algorithm follows to update the state-action Q table to learn an agent with optimal behaviour given the environment.

### Algorithm
q(s,a)= -&infin; for all (s,a)\
v(s)  = 0 for all s
\
\
For each **episode**:
\
&emsp;For each **S<sub>t</sub>** and **A<sub>t</sub>**:\
&emsp;&emsp;take action **A<sub>t</sub>**, observe **S<sub>t+1</sub>** and **R<sub>t+1</sub>**\
&emsp;&emsp;**S<sub>t+1</sub>**=(**Temp<sub>t+1, float</sub>**, **HVAC<sub>t+1</sub>**) where **Temp** is a slowly changing variable\
&emsp;&emsp;**p<sub>0</sub>**= **Temp<sub>t+1, Discrete, high</sub>** - **Temp<sub>t+1, float</sub>**\
&emsp;&emsp;**p<sub>1</sub>**= 1-**p<sub>0</sub>**\
&emsp;&emsp;**v<sub>0</sub>**= v(**Temp<sub>t+1, Discrete, low</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v<sub>1</sub>**= v(**Temp<sub>t+1, Discrete, high</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v(S<sub>t+1</sub>)**= **v<sub>0</sub>** &times; **p<sub>0</sub>** + **v<sub>1</sub>** &times; **p<sub>1</sub>**\
&emsp;&emsp;**q(S<sub>t</sub>, A<sub>t</sub>)** = **R<sub>t+1</sub>** + &gamma;&times;**v(S<sub>t+1</sub>)**\
&emsp;**v**=max(**q**,-1)

The above algorithm is computed in the **dp_long** method of the **agent** module.
