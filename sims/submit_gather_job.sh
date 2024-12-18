#!/bin/bash

#SBATCH --job-name gather_job
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00
#SBATCH --mem=4G
#SBATCH --mail-user=joelne@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/gather_job.out

source ~/virtual_envs/physicell/bin/activate
python -u sims/gather_tens.py > slurm_logs/gather_job.log
