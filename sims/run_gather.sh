#!/bin/bash

if [ "$SYSTEM_ENV" != "laptop" ]; then
		source ~/virtual_envs/physicell/bin/activate
fi

python -u sims/gather_tens.py > logs/gather_job.log
