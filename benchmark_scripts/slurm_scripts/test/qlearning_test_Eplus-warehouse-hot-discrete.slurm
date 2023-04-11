#!/bin/bash
#SBATCH --job-name=QLearning_Test_Eplus-warehouse-hot-discrete

#SBATCH --partition=cpu

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=32G

#SBATCH --output=slurm-%x.out

#SBATCH --error=slurm-%x.err

# prepare your environment here
export PATH=/pkgs/anaconda3/bin:$PATH
source activate $HOME/condaenvs/sinergym

# put your command here
python $HOME/MBRL-HVAC-Energy-Optimization/benchmark_scripts/test.py --model-path "$HOME/MBRL-HVAC-Energy-Optimization/result_models/5tiles/Eplus-warehouse-hot-discrete-train-v1/QLearning" --wandb --num-eval 1