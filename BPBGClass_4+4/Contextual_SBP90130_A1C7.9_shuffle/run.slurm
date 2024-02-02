#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=shuffle
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --output=slurm_log/%x.%j
# SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
# SBATCH --partition=gpu          # Submit job to the gpu queue

# SBATCH --gres=gpu:a100:1
# SBATCH --gres=gpu:rtx:1 
# SBATCH --gres=gpu:t4:1

# execute python script
python Contextual.py

# Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"