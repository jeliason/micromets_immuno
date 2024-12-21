#!/bin/bash

if [ "$SYSTEM_ENV" != "laptop" ]; then
		source ~/virtual_envs/physicell/bin/activate
fi
python -u sims/generate_theta.py --n_samples 64 --seed_start 1234 --num_sims_per_theta 20 > logs/gen_theta.log
