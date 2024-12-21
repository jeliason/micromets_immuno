#!/bin/bash

if [ "$SYSTEM_ENV" != "laptop" ]; then
		source ~/virtual_envs/physicell/bin/activate
fi

python -u sims/run_simulation.py --sim_id 300 > logs/sim.log
