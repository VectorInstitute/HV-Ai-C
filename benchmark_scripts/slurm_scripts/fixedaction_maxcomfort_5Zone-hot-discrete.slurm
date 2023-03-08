#!/bin/bash
#SBATCH --job-name=FixedAction_MaxComfort_5Zone-hot-discrete

#SBATCH --partition=cpu

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=32G

#SBATCH --output=slurm-%x.out

#SBATCH --error=slurm-%x.err

# prepare your environment here
export PATH=/pkgs/anaconda3/bin/conda:$PATH
source activate $HOME/condaenvs/sinergym

# put your command here
python $HOME/MBRL-HVAC-Energy-Optimization/benchmark_scripts/train.py --algo FixedActionMaxComfort --env Eplus-5Zone-hot-discrete-train-v1