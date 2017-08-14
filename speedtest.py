# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:36:52 2017

@author: Max
"""

"""
A simple 2-node simulation demonstration the application of GP-CaKe. Of particular interest are the covariance parameters
that define the constraints on the posterior shape of the causal kernels.

References:
    Ambrogioni, L., Hinne, M., van Gerven, M., & Maris, E. (2017). GP CaKe: Effective brain connectivity with causal kernels,
    pp. 1–10. Retrieved from http://arxiv.org/abs/1705.05603

Last updated on July 6th, 2017.
"""

import numpy as np
#import importlib

"""
Simulation and GP-CaKe packages.
"""

import simulator as sim
import gpcake
import utility


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--processes',
            action="store", dest="num_processes",
            help="Number of parallel processes", default="1")
parser.add_argument('-d', '--density',
            action="store", dest="density",
            help="Expected network density", default="0.2")
parser.add_argument('-ntest', action="store", dest="ntest", default="30")
parser.add_argument('-ntrain', action="store", dest="ntrain", default="30")
parser.add_argument('-n', '--nnodes', action="store", dest="nnodes", default="30")

options             = parser.parse_args()
num_processes       = int(options.num_processes)
ntrials_test        = int(options.ntest)
ntrials_train       = int(options.ntrain)
d                   = float(options.density)
p                   = int(options.nnodes)


"""
Simulation parameters. Here, we construct a 2-node graph with one connection (with max. strength <connection_strength>).
We create a 4 second time series per node, with a sampling rate of 100 Hz.
"""

adj_mat                 = np.zeros((p,p))
for i in range(0,p):
    for j in range(0,p):
        if i != j and np.random.rand() < d:
            adj_mat[i,j] = 1

            
connection_strength     = 1.0
time_step               = 0.01
time_period             = 4.
time_range              = np.arange(-time_period / 2, time_period / 2, time_step)
n                       = int(time_period / time_step)
simulation_params       = {'network'                : adj_mat,
                           'connection_strength'    : connection_strength,
                           'time_step'              : time_step,
                           'time_period'            : time_period}

"""
Simulation settings. We generate <ntrials_train> trials to train the dynamic parameters on,
and <ntrials_test> to learn the GP posterior.
"""


simulation                                          = sim.integroDifferential_simulator()
print('Generating simulation samples')
(training_samples, testing_samples, ground_truth)   = simulation.simulate_network_dynamics(ntrials_train, ntrials_test, simulation_params)

"""
Plot a few samples to see the generated time series.
"""

#utility.plot_samples(training_samples[0:3])

"""
Simulation is done. Time to bake some cake!
"""

cake = gpcake.gpcake()
cake.initialize_time_parameters(time_step, time_period)
cake.dynamic_parameters["number_sources"] = p

"""
Select internal dynamics type. Currently implemented are "Relaxation" and "Oscillation".
"""

cake.dynamic_type = "Relaxation"


"""
Optimize the univariate likelihoods for each node for the dynamic parameters using a grid search.
"""


print('Learning dynamic parameters')
cake.dynamic_parameters = {}
cake.dynamic_parameters["relaxation_constants"] = np.zeros(shape = (p,1))
cake.dynamic_parameters["amplitude"] = np.zeros(shape = (p,1))
cake.dynamic_parameters["moving_average_time_constants"] = 1e15 * np.ones(shape = (p,1))
cake.dynamic_parameters["number_sources"] = p
for i in range(0,p):
    cake.dynamic_parameters['relaxation_constants'][i] = 36
    cake.dynamic_parameters['amplitude'][i] = 0.010

"""
Set the parameters of the causal kernel.
"""

cake.covariance_parameters = {  "time_scale"        : 0.15,     # Temporal smoothing
                                "time_shift"        : 0.05,     # Temporal offset
                                "causal"            : "yes",    # Hilbert transform
                                "spectral_smoothing": np.pi }   # Temporal localization
cake.noise_level = 0.05

"""
Compute the posteriors for each of the p*(p-1) connections.
"""

print('Computing posterior kernels')

        
utility.tic()
cake.parallelthreads=num_processes
connectivity_parallel_flat = cake.run_analysis(testing_samples)
utility.plot_connectivity(ground_truth, connectivity_parallel_flat, time_range, t0=-0.5, filename='speedtest_flat_{:d}processes_{:d}nodes_{:d}trials.pdf'.format(num_processes, p, ntrials_test))
print(utility.toc())

utility.tic()
cake.parallelthreads=num_processes
connectivity_parallel = cake.run_analysis(testing_samples, onlyTrials=True)
utility.plot_connectivity(ground_truth, connectivity_parallel, time_range, t0=-0.5, filename='speedtest_{:d}processes_{:d}nodes_{:d}trials.pdf'.format(num_processes, p, ntrials_test))
print(utility.toc())

utility.tic()
cake.parallelthreads=1
connectivity_serial = cake.run_analysis(testing_samples)
utility.plot_connectivity(ground_truth, connectivity_serial, time_range, t0=-0.5, filename='speedtest_{:d}processes_{:d}nodes_{:d}trials.pdf'.format(1, p, ntrials_test))
print(utility.toc())


