#!/bin/bash

#SBATCH --job-name mm_sim
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --mail-user=joelne@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/slurm_%a.out

source ~/virtual_envs/physicell/bin/activate
python -u sims/run_simulation.py --theta_id $SLURM_ARRAY_TASK_ID --n_samples 2 --seed_start 1234 > slurm_logs/script_$SLURM_ARRAY_TASK_ID.log
