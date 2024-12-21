#!/bin/bash

if [ "$SYSTEM_ENV" != "laptop" ]; then
		source ~/virtual_envs/physicell/bin/activate
fi
python -u sims/generate_theta.py --n_samples 128 --seed_start 1234 --num_sims_per_theta 10 > logs/gen_theta.log
