from scipy.stats import qmc
import numpy as np
import argparse
from src.utils import get_param_dict, get_output_path, expand_grid

OUTPUT_PATH = get_output_path()

def is_power_of_two(n):
		if n <= 0:
				return False
		return (n & (n - 1)) == 0

def main():
		parser = argparse.ArgumentParser()
		parser.add_argument('--n_samples', type=int, default=1000)
		parser.add_argument('--seed_start', type=int, default=1234)
		parser.add_argument('--num_sims_per_theta', type=int, default=0)

		args = parser.parse_args()
		seed_start = args.seed_start
		n_samples = args.n_samples
		num_sims_per_theta = args.num_sims_per_theta

		params_dict = get_param_dict()

		if not is_power_of_two(n_samples):
				raise ValueError("n_samples must be a power of 2")
		prior_min = [x[0] for x in params_dict.values()]
		prior_max = [x[1] for x in params_dict.values()]
		d = len(prior_min)
		sampler = qmc.Sobol(d=d, scramble=False, seed = seed_start)
		sample = sampler.random_base2(m=int(np.log2(n_samples)))
		sample = qmc.scale(sample,prior_min, prior_max)

		# # swap values if min > saturation
		# for i in range(sample.shape[0]):
		# 		params = sample[i]
		# 		print(params)
		# 		if params[1] > params[2]:
		# 				params[1], params[2] = params[2], params[1]
				
		# 		if params[4] > params[5]:
		# 				params[4], params[5] = params[5], params[4]

		# 		sample[i] = params

		np.save(OUTPUT_PATH + 'theta.npy', sample)

		grid = expand_grid(
				num_sims_per_theta=[x for x in range(num_sims_per_theta)],
				theta_id= [x for x in range(n_samples)]
		)

		grid.to_csv(OUTPUT_PATH + 'grid.csv', index=False)

if __name__ == '__main__':
		main()
