#!/bin/bash
#SBATCH --job-name=HNP_QLearning_Train_Eplus-5Zone-hot-discrete

#SBATCH --partition=cpu

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=32G

#SBATCH --output=slurm-%x.out

#SBATCH --error=slurm-%x.err

# prepare your environment here
export PATH=/pkgs/anaconda3/bin:$PATH
source activate $HOME/condaenvs/sinergym

# put your command here
python $HOME/MBRL-HVAC-Energy-Optimization/benchmark_scripts/train.py --algo HNP-QLearning --env Eplus-5Zone-hot-discrete-train-v1