"""Test of DROPO in the gym Hopper environment

For this sim2sim case, a dataset has been collected offline
from the Hopper-v2 gym environment with a semi-converged policy.
The randomized parameters are the four link masses of the Hopper.

Examples:
	
	[Quick test] python3 test_dropo.py --sparse-mode -n 10 -l 1 --budget 1000 -av --epsilon 1e-5 --seed 100 --dataset datasets/hopper10000 --normalize --logstdevs

	[Advanced test] python3 test_dropo.py -n 2 -l 1 --budget 5000 -av --epsilon 1e-5 --seed 100 --dataset datasets/hopper10000 --normalize --logstdevs --now 10

"""
import glob
import sys
import pdb
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

from method.dropo import Dropo
from utils.utils import *
from randommasshopper.random_mass_hopper import *

def main():

	args = parse_args_dropo()
	set_seed(args.seed)

	sim_env = gym.make('RandomMassHopper-v0')

	print('Action space:', sim_env.action_space)
	print('State space:', sim_env.observation_space)
	print('Initial dynamics:', sim_env.get_task())
	print('\nARGS:', vars(args))

	target_task = [3.53429174, 3.92699082, 2.71433605, 5.0893801]   # Ground truth target domain task to learn (in sim2real is not known a priori)
	print('Target task to learn:', target_task)

	observations = np.load(glob.glob(os.path.join(args.dataset, '*_observations.npy'))[0])
	next_observations = np.load(glob.glob(os.path.join(args.dataset, '*_nextobservations.npy'))[0])
	actions = np.load(glob.glob(os.path.join(args.dataset, '*_actions.npy'))[0])
	terminals = np.load(glob.glob(os.path.join(args.dataset, '*_terminals.npy'))[0])

	T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals }

	# Initialize dropo
	dropo = Dropo(sim_env=sim_env,
				  t_length=args.l,
				  scaling=args.scaling,
				  seed=args.seed,
				  sync_parall=(not args.no_sync_parall))


	# Load target offline dataset
	dropo.set_offline_dataset(T, n=args.n_trajectories, sparse_mode=args.sparse_mode)


	# Run DROPO
	(best_bounds,
	 best_score,
	 elapsed,
	 learned_epsilon) = dropo.optimize_dynamics_distribution(opt=args.opt,
												  		   budget=args.budget,
													       additive_variance=args.additive_variance,
													       epsilon=args.epsilon,
													       sample_size=args.sample_size,
													       now=args.now,
													       learn_epsilon=args.learn_epsilon,																					  
													       normalize=args.normalize,
													       logstdevs=args.logstdevs)
	
	

	"""
		OUTPUT RESULTS
	"""

	print('\n-----------')
	print('RESULTS\n')
	
	print('ARGS:', vars(args), '\n\n')

	print('Best means and st.devs:\n---------------')
	print(dropo.pretty_print_bounds(best_bounds),'\n')

	if learned_epsilon is not None:
		print('Best epsilon:', learned_epsilon)

	print('Best score (log likelihood):', best_score)

	if args.sparse_mode:
		print('MSE:', dropo.MSE(dropo.get_means(best_bounds)))
	else:
		print('MSE:', dropo.MSE_trajectories(dropo.get_means(best_bounds)))

	print('Squared distance between best means and target task:', np.linalg.norm(target_task - dropo.get_means(best_bounds)))
	print('Elapsed:', round(elapsed/60, 4), 'min')
 

	if not args.no_output:	# Output results to file
		make_dir(args.output_dir)

		with open(os.path.join(args.output_dir, '')+'dropo_n'+str(args.n_trajectories)+'_l'+str(args.l)+'_'+datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.txt', 'a', encoding='utf-8') as file:
			
			print('-----------', file=file)
			print('RESULTS\n', file=file)
			
			print('ARGS:', vars(args), '\n\n', file=file)

			print('Best means and st.devs:\n---------------', file=file)
			print(dropo.pretty_print_bounds(best_bounds),'\n', file=file)

			if learned_epsilon is not None:
				print('Best epsilon:', learned_epsilon, file=file)

			print('Best score (log likelihood):', best_score, file=file)

			if args.sparse_mode:
				print('MSE:', dropo.MSE(dropo.get_means(best_bounds)), file=file)
			else:
				print('MSE:', dropo.MSE_trajectories(dropo.get_means(best_bounds)), file=file)

			print('Squared distance between best means and target task:', np.linalg.norm(target_task - dropo.get_means(best_bounds)), file=file)
			print('Elapsed:', round(elapsed/60, 4), 'min', file=file)


	return



if __name__ == '__main__':
	main()