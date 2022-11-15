from argparse import ArgumentParser
import os

import numpy as np

def parse_args_dropo():
	parser = ArgumentParser()

	# RECOMMENDED FLAGS
	parser.add_argument("--additive_variance", "-av", default=False, action='store_true', help="RECOMMENDED. Add value --epsilon to the diagonal of the cov_matrix to regularize the next-state distribution inference (default: False)")
	parser.add_argument("--normalize", default=False, action='store_true', help="RECOMMENDED. Normalize dynamics search space to [0,4] as a regularization for CMA-ES.")
	parser.add_argument("--logstdevs", default=False, action='store_true', help="RECOMMENDED. Optimize stdevs in log space. (Default: false)")

	# Hyperparameters
	parser.add_argument("--n-trajectories", "-n", type=int, default=None, help="Number of target trajectories for running DROPO. if --sparse-mode is selected, this parameter refers to the number of single TRANSITIONS instead.")
	parser.add_argument("-l", type=int, default=1, help="Lambda hyperparameter.")
	parser.add_argument("--epsilon", "-eps", type=float, default=1e-3, help="RECOMMENDED. Epsilon hyperparameter. Valid only when --additive_variance is set (default: 1e-3)")
	parser.add_argument('--env', default='RandomHopper-v0', type=str, help='Gym-registered environment.')
	parser.add_argument("--output-dir", type=str, default='output', help="Output directory for results")
	parser.add_argument("--scaling", default=False, action='store_true', help="Scaling each state dimension (Default: False)")
	parser.add_argument("--now", type=int, default=1, help="Number of workers for parallelization (Default: 1 => no parallelization)")
	parser.add_argument("--seed", type=int, default=0, help="Set a specific seed")
	parser.add_argument("--opt", type=str, default='cma', help="nevergrad optimizer [oneplusone, bayesian, twopointsde, pso, tbpsa, random, meta, cma (default)]")
	parser.add_argument("--no-output", "-no", default=False, action='store_true', help="If set, DO NOT save the output of optimization problem to --output-dir")
	parser.add_argument("--budget", type=int, default=1000, help="Number of evaluations in the opt. problem (Default: 1000)")
	parser.add_argument("--sample_size", "-ss", type=int, default=100, help="Number of observations to sample to estimate the next-state distribution (Default: 100)")
	parser.add_argument("--dataset", type=str, default='datasets/hopper10000', help="Specify directory containing a custom dataset to use.")
	parser.add_argument("--sparse-mode", "-sm", default=False, action='store_true', help="Whether to use sparse transitions for running DROPO than reproducing full episodes. (Default: False)")
	parser.add_argument("--no-sync-parall", default=False, action='store_true', help="If set, avoids asking `popsize` values before telling their values during parallelization.")
	
	# Not officially supported
	parser.add_argument("--learn-epsilon", default=False, action='store_true', help="(Not recommended) Whether to learn the hyperparameter --epsilon (default: False)")

	args = parser.parse_args()

	return args


def set_seed(seed):
	if seed > 0:
		np.random.seed(seed)


def make_dir(dir_path):
	try:
		os.mkdir(dir_path)
	except OSError:
		pass

	return