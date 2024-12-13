#!/bin/bash

#SBATCH --job-name mm_sim
#SBATCH --array=0-49
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00
#SBATCH --mem=2G
#SBATCH --mail-user=joelne@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/slurm_%a.out

python -u sims/run_simulation.py --theta_id $SLURM_ARRAY_TASK_ID --n_samples 50 --seed 1234 > slurm_logs/script_$SLURM_ARRAY_TASK_ID.log
