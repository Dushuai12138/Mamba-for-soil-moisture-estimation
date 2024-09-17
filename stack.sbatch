#!/bin/bash

#SBATCH --partition=h240
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00

cd $SLURM_SUBMIT_DIR

python stack_bands.py

