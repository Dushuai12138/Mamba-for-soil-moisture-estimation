#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=LSTM
#SBATCH --output=slurm-%j-LSTM.out

cd $SLURM_SUBMIT_DIR

python train.py --model_name LSTM
