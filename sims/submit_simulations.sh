#!/bin/bash

#SBATCH --job-name mm_sim
#SBATCH --array=0-1279%100 # 1280 total simulations, 100 at a time
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2G
#SBATCH --mail-user=joelne@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/sim_%a.out

source ~/virtual_envs/physicell/bin/activate
python -u sims/run_simulation.py --sim_id $SLURM_ARRAY_TASK_ID > logs/sim_$SLURM_ARRAY_TASK_ID.log
