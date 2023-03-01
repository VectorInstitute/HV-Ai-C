#!/bin/bash
#SBATCH --job-name=HNP_QLearning_Train

#SBATCH --partition=cpu

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=32G

#SBATCH --output=slurm-%x.out

#SBATCH --error=slurm-%x.err

# prepare your environment here
export PATH=/pkgs/anaconda3/bin/conda:$PATH
conda activate $HOME/condaenvs/sinergym

# put your command here
python $HOME/MBRL-HVAC-Energy-Optimization/benchmark_scripts/main.py $HOME/MBRL-HVAC-Energy-Optimization/benchmark_scripts/configs/HNP_QLearning_config.yaml