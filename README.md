English | [Français](#apprentissage-par-renforcement-basé-sur-un-modèle-pour-un-contrôle-cvc-écoénergétique-dans-les-centres-de-données)

# Model-Based Reinforcement Learning for Energy Efficient Data Centre HVAC Control

This repository introduces a reference solution for training a **Model-Based Reinforcement Learning** (MBRL) agent to control the HVAC system in a small-room datacentre such that **energy utilization is minimized**. When this agent was piloted in a small site, cooling savings of 18% (excluding IT load) and heating savings of 10% (excluding IT load) were realized, for a combined estimated annual electricity savings of 11.5%.  Therefore, potentially this agent can be implemented to reduce electricity costs as well as harmful greenhouse gas emissions that contribute to climate change.

# Repository Organization

All solution files are located in the `src` directory, which are organized as follows:

* `agent.py`: Reinforcement learning agent using the Hyperspace Neighbour Penetration algorithm.
* `environment.py`: OpenAI Gym environment simulating a room with HVAC controls. Parameterized by a file `configs/environment_params.cfg`, which must be populated according to instructions below, under **State Transition Parameters**.
* `util.py`: Utility functions supporting training.
* `main.py`: Main training loop for the reinforcement learning agent.

# Dependencies

* Python ≥ 3.9
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
The possible set of actions from a given state are:\
\
**["action_idle", "action_compressor_on", "action_freecool_on", "action_heater_on"]**\
\
It should be noted, however, that not all actions are possible at every state; for instance, if we are at or near a maximum allowable temperature, we cannot select "action_heater_on". Such illegal state-action pairs are not allowed during training.

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
&emsp;&emsp;**p<sub>0</sub>**= (**Temp<sub>t+1, Discrete, high</sub>** - **Temp<sub>t+1, float</sub>**)/(**Temp<sub>t+1, Discrete, high</sub>** - **Temp<sub>t+1, Discrete, low</sub>**)  where the denominator is the unit size of **Temp<sub>Discrete</sub>**.\
&emsp;&emsp;**p<sub>1</sub>**= 1-**p<sub>0</sub>**\
&emsp;&emsp;**v<sub>0</sub>**= v(**Temp<sub>t+1, Discrete, low</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v<sub>1</sub>**= v(**Temp<sub>t+1, Discrete, high</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v(S<sub>t+1</sub>)**= **v<sub>0</sub>** &times; **p<sub>0</sub>** + **v<sub>1</sub>** &times; **p<sub>1</sub>**\
&emsp;&emsp;**q(S<sub>t</sub>, A<sub>t</sub>)** = **R<sub>t+1</sub>** + &gamma;&times;**v(S<sub>t+1</sub>)**\
&emsp;**v**=max(**q**,-1)

The above algorithm is computed in the **dp_long** method of the **agent** module.

---

[English](#model-based-reinforcement-learning-for-energy-efficient-data-centre-hvac-control) | Français

# Apprentissage par renforcement basé sur un modèle pour un contrôle CVC écoénergétique dans les centres de données

Ce répertoire présente une solution de référence faisant appel à l’apprentissage par renforcement basé sur un modèle (MBRL) pour entraîner un agent à contrôler le système CVC dans un centre de données de petite taille de façon à minimiser l’utilisation de l’énergie. Lorsque cet agent a été testé sur un petit site, des économies de refroidissement de 18 % (hors charge informatique) et de chauffage de 10 % (hors charge informatique) ont été réalisées, pour une économie annuelle combinée d’électricité estimée à 11,5 %. Par conséquent, cet agent peut potentiellement être mis en œuvre pour réduire les coûts d’électricité ainsi que les émissions de gaz à effet de serre nocifs qui contribuent aux changements climatiques.

# Organisation du répertoire
Tous les fichiers de la solution sont situés dans le dossier src et organisés de la façon suivante :

* `agent.py` : Agent d’apprentissage par renforcement utilisant l’algorithme HNP (Hyperspace Neighbour Penetration).
* `environment.py` : Environnement OpenAI Gym simulant une pièce avec des contrôles CVC. Il est paramétré par un fichier `configs/environment_params.cfg` qui doit être rempli selon les instructions ci-dessous, sous **Paramètres de transition d’état**.
* `util.py` : Fonctions utilitaires à l’appui de l’entraînement.
* `main.py` : Boucle d’entraînement principale pour l’agent d’apprentissage par renforcement.

# Dépendances
* Python version 3.9 ou supérieure
* [OpenAI Gym](https://github.com/openai/gym)
* [NumPy](https://github.com/numpy/numpy)
* [Pandas](https://github.com/pandas-dev/pandas)

# Aperçu de la solution
Le contrôle de la température des centres de données est un processus essentiel pour maintenir une qualité de service élevée. Il est essentiel de maintenir la température à l’intérieur d’une plage appropriée, mais il est aussi important de considérer que chaque stratégie de chauffage ou de refroidissement comporte des coûts environnementaux et économiques. Par exemple, une stratégie de refroidissement qui fait un usage excessif ou inutile de la climatisation consommera plus d’énergie qu’une stratégie qui exploite mieux le refroidissement naturel. Le refroidissement naturel consiste à contourner le refroidissement mécanique par détente directe en échangeant la chaleur avec l’air extérieur à plus basse température, ce qui en fait une approche plus économique et plus écoénergétique. Tant que les contraintes opérationnelles sont satisfaites, les possibilités de découvrir des solutions de contrôle de la température qui minimisent la consommation d’énergie sont très précieuses.

Ce répertoire fournit une solution de référence pour former un agent MBRL qui contrôlera la température dans un centre de données de petite taille. L’algorithme tire parti d’une innovation dans la conception de la représentation d’état, appelée HNP (Hyperspace Neighbour Penetration). Celle-ci permet à un algorithme classique de programmation dynamique basé sur l’apprentissage Q, ou Q-learning, d’apprendre rapidement des politiques efficaces et écoénergétiques pour le contrôle des systèmes CVC.

Une hypothèse clé de l’algorithme HNP est que les représentations d’état continues avec des transitions très incrémentielles et localement linéaires peuvent être discrétisées efficacement en plages ou bandes de valeurs relativement grossières sous forme de carreaux. Lorsque ce type de discrétisation grossière est appliqué dans un cadre conventionnel d’apprentissage Q, cela peut conduire à des situations où une action (par exemple, la mise en marche d’un compresseur de refroidissement) entraîne un changement si faible dans l’observation de l’état subséquent qu’aucune transition d’état discrète ne s’est réellement produite. Une solution naïve pourrait être d’augmenter la granularité de la représentation d’état, c’est-à-dire de considérer des changements de température beaucoup plus petits comme des éléments distincts de la représentation d’état, mais cela rend l’apprentissage Q basé sur la programmation dynamique intraitable. L’algorithme HNP calcule aussi plusieurs valeurs à partir des frontières des carreaux, puis les agrège en utilisant une norme pondérée. Ainsi, les paires état-action aboutissent à des étapes vers d’autres états et leurs valeurs correspondantes. Cela réduit fortement la taille de l’espace de recherche effectif pour la programmation dynamique et rend le problème plus facile à calculer.

L’algorithme HNP est entièrement décrit dans son article [fondateur](https://arxiv.org/pdf/2106.05497.pdf) et l’algorithme d’apprentissage Q est présenté plus en détail ci-dessous.

# Détails de l’algorithme d’apprentissage par renforcement

## Représentation d’état-action
Les variables d’état disponibles et leurs valeurs correspondantes pour l’environnement sont les suivantes :

### Température
Cette variable d’état suit la température de l’environnement entre les valeurs définies **[low, high]** avec une taille de pas donnée.

### Chauffage, ventilation et climatisation
Cette variable d’état suit l’état du système CVC avec les valeurs discrètes possibles de :
 
**["status_idle", "status_compressor_on", "status_freecool_on", "status_heater_on"]**

### Actions
Les actions possibles à partir d’un état donné sont :
 
**["action_idle", "action_compressor_on", "action_freecool_on", "action_heater_on"]**
 
Il faut noter que toutes les actions ne sont pas possibles dans tous les états. Par exemple, à la température maximale admissible ou proche de celle-ci, l’action « action_heater_on » ne peut pas être sélectionnée. De telles paires état-action illégales ne sont pas permises durant l’entraînement.

### Récompenses
Il y a trois types différents de récompenses négatives associées à l’environnement de contrôle de la température :
 
**1. Consommation d’énergie normale :** Cette récompense pénalise l’agent pour la consommation d’énergie.
**2. Récompense pour le démarrage d’un appareil :** Récompense associée au démarrage du compresseur ou de l’appareil de refroidissement naturel. Notez que le démarrage des appareils nécessite plus d’énergie que la sélection des actions correspondantes et qu’il s’agit d’une récompense unique.
**3. Violation de la température :** Pénalisation du passage à un état non admissible, comme une température supérieure au maximum défini.
À chaque transition d’état, la récompense correspondante est une somme des trois types de récompenses énoncés.

## Paramètres de transition d’état

Il y a cinq transitions possibles correspondant aux quatre actions possibles :
**["compressor", "freecool", "heater", "idle_above_th", "idle_below_th"]**
**« idle_above_th »** et **« idle_below_th »** représentent deux situations d’inactivité, où **« above_th »** indique que la température intérieure est supérieure au seuil et **« below_th »** indique que la température intérieure est égale ou inférieure au seuil.

Une régression linéaire est utilisée pour calculer le taux de changement de température (par minute par degré) pour chaque type de transition. La différence entre la température extérieure actuelle et la température intérieure actuelle est utilisée comme variable explicative, et la différence entre la température intérieure de la minute suivante et la température intérieure actuelle est utilisée comme variable dépendante. Le gradient (**coef**) et le point d’interception (**intercept**) de la droite d’ajustement sont enregistrés pour une utilisation ultérieure.

Trois lignes différentes sont ajustées pour chaque transition :

**["pos_diff", "neg_diff", "all"]**

Alors que **« pos_diff »** est appliqué lorsque la température extérieure est supérieure à la température intérieure, **« neg_diff »** est appliqué lorsque la température extérieure est égale ou inférieure à la température intérieure, et **« all »** ignore la différence de température extérieure intérieure.

La valeur de ces paramètres doit être enregistrée dans `environment_params.cfg`.

**Remarque :** Le code permettant de calculer les paramètres requis pour le fichier `environment_params.cfg` au moyen de données historiques de CVC _n’est pas inclus_ dans la version initiale de cette solution de référence, puisque la version actuelle n’a pas été adaptée pour être généralisée en dehors de l’environnement de TELUS. L’approche est facile à reproduire au moyen des instructions ci-dessus, et le code de référence sera fourni dans une version ultérieure.

# Programmation HNP
Voici la série d’étapes que suit l’algorithme de programmation dynamique pour mettre à jour la table état-action Q afin de former un agent au comportement optimal compte tenu de l’environnement.

### Algorithme
q(s,a)= -&infin; for all (s,a)\
v(s)  = 0 for all s
\
\
For each **episode**:
\
&emsp;For each **S<sub>t</sub>** and **A<sub>t</sub>**:\
&emsp;&emsp;take action **A<sub>t</sub>**, observe **S<sub>t+1</sub>** and **R<sub>t+1</sub>**\
&emsp;&emsp;**S<sub>t+1</sub>**=(**Temp<sub>t+1, float</sub>**, **HVAC<sub>t+1</sub>**) where **Temp** is a slowly changing variable\
&emsp;&emsp;**p<sub>0</sub>**= (**Temp<sub>t+1, Discrete, high</sub>** - **Temp<sub>t+1, float</sub>**)/(**Temp<sub>t+1, Discrete, high</sub>** - **Temp<sub>t+1, Discrete, low</sub>**)  where the denominator is the unit size of **Temp<sub>Discrete</sub>**.\
&emsp;&emsp;**p<sub>1</sub>**= 1-**p<sub>0</sub>**\
&emsp;&emsp;**v<sub>0</sub>**= v(**Temp<sub>t+1, Discrete, low</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v<sub>1</sub>**= v(**Temp<sub>t+1, Discrete, high</sub>**, **HVAC<sub>t+1</sub>**)\
&emsp;&emsp;**v(S<sub>t+1</sub>)**= **v<sub>0</sub>** &times; **p<sub>0</sub>** + **v<sub>1</sub>** &times; **p<sub>1</sub>**\
&emsp;&emsp;**q(S<sub>t</sub>, A<sub>t</sub>)** = **R<sub>t+1</sub>** + &gamma;&times;**v(S<sub>t+1</sub>)**\
&emsp;**v**=max(**q**,-1)

L’algorithme ci-dessus est calculé selon la méthode **dp_long** du module d’**agent**.

