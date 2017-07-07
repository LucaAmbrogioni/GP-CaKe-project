# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# <codecell>

def get_gaussian_random_vector(covariance, shape):
    mean = np.zeros(shape = (max(shape),))
    random_array = np.random.multivariate_normal(mean, covariance)
    random_vector = np.matrix(np.reshape(random_array, newshape = shape))
    return random_vector

# <codecell>

def get_AR_sample(number_time_points, coefficients, initial_state, innovation_covariance):
    max_time_lag = len(coefficients)
    process_shape = (initial_state).shape
    AR_process = [initial_state]
    for time_point in range(0, number_time_points):
        new_value = get_gaussian_random_vector(innovation_covariance, process_shape)
        for time_lag in range(1, min(max_time_lag, time_point)):
            new_value += coefficients[time_lag]*AR_process[-time_lag]
        AR_process += [new_value]
    return np.transpose(np.array(np.bmat(AR_process)))

# <codecell>

def generate_AR_samples(parameters, number_samples):
    number_time_points = parameters["number_time_points"]
    coefficients = parameters["coefficients"]
    initial_state = parameters["initial_state"]
    innovation_covariance = parameters["innovation_covariance"]
    
    AR_samples = []
    for sample_index in range(0,number_samples):
        AR_samples += [get_AR_sample(number_time_points, coefficients, initial_state, innovation_covariance)]
    return AR_samples

# <codecell>

def get_coefficients(max_time_lag, number_processes, autocoefficients, simulation_type, simulation_parameters):
    coefficients = []
    for time_lag in range(0, max_time_lag):
        lagged_coefficients = np.matrix(np.zeros(shape = (number_processes, number_processes)))
        for first_process in range(0, number_processes):
            for second_process in range(0, number_processes):
                if first_process == second_process:
                    lagged_coefficients[first_process, second_process] = autocoefficients[first_process][time_lag]
                else:
                    if simulation_type is "smooth":
                        weights_matrix = simulation_parameters["weights_matrix"]
                        connectivity_function = simulation_parameters["connectivity_function"]
                        lagged_coefficients[first_process, second_process] = weights_matrix[first_process, second_process]*connectivity_function(time_lag)
                    elif simulation_type is "sparse":
                        lags_matrix = simulation_parameters["lags_matrix"]
                        weights_matrix = simulation_parameters["weights_matrix"]
                        if time_lag == lags_matrix[first_process, second_process]:
                            lagged_coefficients[first_process, second_process] = weights_matrix[first_process, second_process]
                    elif simulation_type is "random":
                        coefficient_sd = simulation_parameters["coefficient_sd"]
                        weights_matrix = simulation_parameters["weights_matrix"]
                        if time_lag == 0:
                            lagged_coefficients[first_process, second_process] = 0.
                        else:
                            lagged_coefficients[first_process, second_process] = weights_matrix[first_process, second_process]*np.random.normal(loc=0.0, scale=coefficient_sd)
        coefficients += [lagged_coefficients]
    return coefficients

# <codecell>

def get_autocoefficients(number_processes, max_time_lag, AR1_coefficient):
    autocoefficients = []
    for process in range(0, number_processes):
        process_autocoefficients = max_time_lag*[0]
        process_autocoefficients[1] = AR1_coefficient
        autocoefficients += [process_autocoefficients]
    return autocoefficients
                  

# <codecell>

# script
#parameters = {}
#parameters["number_time_points"] = 500
#number_processes = 2
 
##
#max_time_lag = 10
#number_processes = 3
#simulation_type = 'random'
#autocoefficients = get_autocoefficients(number_processes, max_time_lag, 0.9)
#simulation_parameters = {}
#simulation_parameters["lags_matrix"] = np.matrix([[0,9],[1,0]])
#simulation_parameters["weights_matrix"] = np.matrix([[0,0.1],[0,0]])
#coefficients = get_coefficients(max_time_lag, number_processes, autocoefficients, simulation_type, simulation_parameters)
#simulation_parameters["coefficient_sd"] = 1.
#simulation_parameters["weights_matrix"] = np.matrix([[0,0.1, 0],[0,0, 0.1], [0,0, 0]])
#parameters["coefficients"] = get_coefficients(max_time_lag, number_processes, autocoefficients, simulation_type, simulation_parameters)
##
#parameters["innovation_covariance"] = np.identity(number_processes)
#parameters["initial_state"] = get_gaussian_random_vector(covariance = parameters["innovation_covariance"], 
#                                                         shape = (number_processes,1))
#AR_samples = generate_AR_samples(parameters = parameters, 
#                                 number_samples = 10)

# <codecell>

#plt.plot(AR_samples[3])

# <codecell>


