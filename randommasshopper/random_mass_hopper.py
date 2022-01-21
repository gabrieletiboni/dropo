"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class HopperEnvMassOnly(MujocoEnv, utils.EzPickle):
    def __init__(self, init_task=None):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}
        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])

        self.task_dim = self.original_masses.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass'}
        self.sampling = None


    def get_task_string(self):
        masses = ", ".join([f"{n} mass: {m:.2f}" for m, n in zip(self.get_task(), self.sim.model.body_names[1:])])
        return masses

    def set_task_search_bounds(self):
        """Sets the task search bounds based on how they are specified in get_search_bounds_mean"""
        
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """

        search_bounds_mean = {
               'torsomass': (0.5, 8.0),
               'thighmass': (0.5, 8.0),
               'legmass': (0.5, 8.0),
               'footmass': (0.5, 8.0),
        }
        return search_bounds_mean[self.dynamics_indexes[index]]

    def set_random_task(self):
        self.set_task(*self.sample_task())

    def sample_task(self):
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        elif self.sampling == 'truncnorm':
            a,b = -2, 2
            sample = []

            for mean, std in zip(self.mean_task, self.stdev_task):
                
                # Assuming all parameters > 0.1
                attempts = 0
                obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                while obs < 0.1:
                    obs = truncnorm.rvs(a, b, loc=mean, scale=std)

                    attempts += 1
                    if attempts > 20:
                        raise Exception('Not all samples were above > 0.1 after 20 attempts')

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'gaussian':
            sample = []

            for mean, std in zip(self.mean_task, self.stdev_task):

                # Assuming all parameters > 0.1
                attempts = 0
                obs = np.random.randn()*std + mean
                while obs < 0.1:
                    obs = np.random.randn()*std + mean

                    attempts += 1
                    if attempts > 20:
                        raise Exception('Not all samples were above > 0.1 after 20 attempts')

                sample.append( obs )

            return np.array(sample)
        else:
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr from file.')

        return

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_task(self, *task):
        self.sim.model.body_mass[1:] = task

    def set_udr_distribution(self, bounds):
        self.sampling = 'uniform'
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i*2]
            self.max_task[i] = bounds[i*2 + 1]
        return

    def set_truncnorm_distribution(self, bounds):
        self.sampling = 'truncnorm'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def set_gaussian_distribution(self, bounds):
        self.sampling = 'gaussian'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def load_dr_distribution_from_file(self, filename):
        dr_type = None
        bounds = None

        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            dr_type = str(next(reader)[0])
            bounds = []

            second_row = next(reader)
            for col in second_row:
                bounds.append(float(col))

        if dr_type is None or bounds is None:
            raise Exception('Unable to read file:'+str(filename))

        if len(bounds) != self.task_dim*2:
            raise Exception('The file did not contain the right number of column values')

        if dr_type == 'uniform':
            self.set_udr_distribution(bounds)
        elif dr_type == 'truncnorm':
            self.set_truncnorm_distribution(bounds)
        elif dr_type == 'gaussian':
            self.set_gaussian_distribution(bounds)
        else:
            raise Exception('Filename is wrongly formatted: '+str(filename))

        return


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_full_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        return mjstate

    def get_initial_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        return mjstate

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


    # Debug
    def is_done(self):
        posafter, height, ang = self.sim.data.qpos[0:3]

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        
        return done


gym.envs.register(
        id="RandomMassHopper-v0",
        entry_point="%s:HopperEnvMassOnly" % __name__,
        max_episode_steps=500,
)

