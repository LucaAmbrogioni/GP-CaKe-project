
# coding: utf-8

# In[2]:

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special as functions
import scipy.signal as sig_tools
import autoregressive_simulations as AR
import statsmodels.tsa.vector_ar.var_model as VAR


# In[3]:

def matrix_division(divider, divided, side, cholesky):
    X = np.matrix(divided)
    if cholesky is "yes":
        M = np.matrix(np.linalg.cholesky(divider))
        if side is "right":
            first_division = np.linalg.solve(M,X.T).T
            result = np.linalg.solve(M.T,first_division.T).T
        elif side is "left":
            first_division = np.linalg.solve(M,X)
            result = np.linalg.solve(M.T,first_division)
        else:
            print "The side should be either left or right"
            return
    else:
        M = np.matrix(divider)
        if side is "right":
            result = np.linalg.solve(M.T,X.T).T
        elif side is "left":
            result = np.linalg.solve(M,X)
        else:
            print "The side should be either left or right"
            return
    return result


# In[4]:

class integroDifferential_simulator(object):
    #  
    def __init__(self):
        self.dynamic_type = "Relaxation"
        self.dynamic_parameters = {"number_sources": 2,
                                   "connectivity_weigths": np.array([[0, 1],[0, 0]]),
                                   "connectivity_relaxations_constants": np.array([[0, 10],[10, 0]]),
                                   "moving_average_time_constants": np.array([100, 100]),
                                   "relaxation_constants": np.array([10, 10])}
        self.time_parameters = {"time_period": 4.,
                                "time_step": 0.01}
        self.time_meshgrid = []
        self.samples = []
    #
    def get_time_meshgrid(self, time_period, time_step):
        time_range = np.arange(-time_period/2., time_period/2., time_step)          
        time_mesh_x, time_mesh_y = np.meshgrid(time_range, time_range)
        time_difference_matrix = (time_mesh_y - time_mesh_x)
        causal_matrix = 0.5*(np.sign(time_difference_matrix) + 1)
        np.fill_diagonal(causal_matrix,0)
        number_time_points = len(time_range)
        return {"time_difference_matrix": time_difference_matrix,
                "causal_matrix":          causal_matrix, 
                "time_step":              time_step,
                "time_range":             time_range,
                "number_time_points":     number_time_points}
    #
    def __get_green_matrices(self, relaxation_constants, time_meshgrid):
        green_matrices = []
        for constant in relaxation_constants:
            green_matrices += [np.matrix(np.multiply(time_meshgrid["causal_matrix"], np.exp(-constant*time_meshgrid["time_difference_matrix"])))*time_meshgrid["time_step"]]
        return green_matrices
    #
    def __get_moving_average_matrices(self, moving_average_time_constants, time_meshgrid):
        moving_average_matrices = []
        for constant in moving_average_time_constants:
            moving_average_matrices += [np.matrix(np.identity(time_meshgrid["number_time_points"])) - (time_meshgrid["time_step"]/constant)*np.matrix(np.multiply(time_meshgrid["causal_matrix"], np.exp(-time_meshgrid["time_difference_matrix"]/constant)))]
        return moving_average_matrices
    #
    def __get_kernel_matrices(self, connectivity_relaxation_constants, connectivity_weights, time_meshgrid):
        kernel_matrices = []
        number_sources = len(connectivity_relaxation_constants)
        for output_index in range(0,number_sources):
            kernel_matrices_row = []
            for input_index in range(0,number_sources):
                nonNormalized_kernel_matrix = np.matrix(np.multiply(np.multiply(time_meshgrid["causal_matrix"], np.exp(-connectivity_relaxation_constants[input_index][output_index]*time_meshgrid["time_difference_matrix"])), time_meshgrid["time_difference_matrix"]))*time_meshgrid["time_step"]
                kernel_matrices_row += [connectivity_weights[input_index][output_index]*nonNormalized_kernel_matrix/np.max(nonNormalized_kernel_matrix)]
            kernel_matrices += [kernel_matrices_row]
        return kernel_matrices
    #
    def __get_block_matrices(self, green_matrices, kernel_matrices, moving_average_matrices):
        block_kernel_matrix = np.bmat(kernel_matrices)
        block_green_matrix = sp.linalg.block_diag(*green_matrices)
        block_moving_average_matrix = sp.linalg.block_diag(*moving_average_matrices)                                
        return {"block_kernel_matrix": block_kernel_matrix, 
                "block_green_matrix": block_green_matrix, 
                "block_moving_average_matrix": block_moving_average_matrix}
    #
    def __get_sample(self, block_matrices, number_sources, number_time_points):
        number_points = block_matrices["block_green_matrix"].shape[0]
        operator = matrix_division(divider = np.identity(number_points) - block_matrices["block_green_matrix"]*block_matrices["block_kernel_matrix"]*block_matrices["block_moving_average_matrix"], 
                                   divided = block_matrices["block_green_matrix"], 
                                   side = "left", 
                                   cholesky = "no")
        driving_noise = np.matrix(np.random.normal(loc=0.0, scale=1.0, size=(number_points, 1)))
        sample = np.reshape(operator*driving_noise, newshape = (number_sources, number_time_points))
        return sample
    #
    def run_sampler(self, number_samples):
        self.time_meshgrid = self.get_time_meshgrid(self.time_parameters["time_period"], 
                                                    self.time_parameters["time_step"])
        green_matrices = self.__get_green_matrices(self.dynamic_parameters["relaxation_constants"], 
                                                   self.time_meshgrid)
        kernel_matrices = self.__get_kernel_matrices(self.dynamic_parameters["connectivity_relaxations_constants"], 
                                                     self.dynamic_parameters["connectivity_weigths"], 
                                                     self.time_meshgrid)
        moving_average_matrices = self.__get_moving_average_matrices(self.dynamic_parameters["moving_average_time_constants"], 
                                                                     self.time_meshgrid)
        block_matrices = self.__get_block_matrices(green_matrices, 
                                                   kernel_matrices,
                                                   moving_average_matrices)
        self.samples = []
        for sample_index in range(0, number_samples):
            self.samples += [self.__get_sample(block_matrices, 
                                               self.dynamic_parameters["number_sources"],
                                               self.time_meshgrid["number_time_points"])]
    #
    def plot_sample(self, sample_index):
        plt.plot(self.time_meshgrid["time_range"], np.transpose(self.samples[sample_index]))
        
        
    


# In[11]:

class GPconnectivityAnalysis(object):
    #
    def __init__(self):
        self.simulator = integroDifferential_simulator()
        self.dynamic_type = []
        self.number_sources = []
        self.time_parameters = []
        self.frequency_range = []
        self.covariance_parameters = {"time_scale": 0.4,
                                      "spectral_smoothing": 4*np.pi*2*10**-1,
                                      "causal": "yes"}
        self.dynamic_parameters = []
        self.noise_level = 1
        self.samples = []
        self.connectivity_kernels_list = []
        self.connectivity_statistics = {"mean": [], 
                                        "variance": []}
    #
    def initialize_time_parameters(self, time_step, time_period):
        self.simulator.time_parameters = {"time_period": time_period,
                                          "time_step": time_step}
        self.simulator.time_meshgrid = self.simulator.get_time_meshgrid(self.simulator.time_parameters["time_period"], 
                                                                        self.simulator.time_parameters["time_step"])
        self.time_parameters = self.simulator.time_parameters
        self.time_parameters.update(self.simulator.time_meshgrid)
        return
    #
    def extract_block(self, matrix, first_indices, second_indices):
        reduced_matrix = matrix[first_indices,:]
        block = reduced_matrix[:,second_indices]
        return np.matrix(block)
    #
    def __get_fourier_time_series(self, time_domain_samples):
        fourier_time_series = []
        number_sources = time_domain_samples.shape[0]
        for index in range(0, number_sources):
            fourier_time_series += [np.fft.fftshift(np.fft.fft(time_domain_samples[index, :]))]
        return fourier_time_series   
    #
    def __get_observation_models(self, fourier_time_series, moving_average_kernels):
        observation_models = []
        index = 0
        for series in fourier_time_series:
            observation_models += [np.matrix(np.diag(np.multiply(moving_average_kernels[index], fourier_time_series[index]).flatten()))]
            index += 1
        return observation_models
    #
    def __get_modified_processes(self, fourier_time_series, dynamic_polynomials):
        modified_processes = []
        index = 0
        for series in fourier_time_series:
            modified_processes += [np.multiply(dynamic_polynomials[index], fourier_time_series[index])]
            index += 1
        return modified_processes
    #
    def __get_covariance_matrix(self):
        time_scale = self.covariance_parameters["time_scale"]
        time_shift = self.covariance_parameters["time_shift"]
        spectral_smoothing = self.covariance_parameters["spectral_smoothing"]
        number_time_points = self.time_parameters["number_time_points"]
        # Frequency grid
        freq_grid_x, freq_grid_y = np.meshgrid(self.frequency_range, self.frequency_range)
        # Stationary part
        stationary_covariance_function = lambda f,l,tau: np.exp(-1j*tau*f-f**2/(2*l**2))
        spectral_width = 2*np.pi*(1/time_scale)
        diagonal_covarianbce_matrix = np.matrix(np.diag(stationary_covariance_function(self.frequency_range, spectral_width, time_shift)))
        # Nonstationary part
        smoothing_covariance_matrix = np.matrix(self.__squared_exponential_covariance(freq_grid_x - freq_grid_y, spectral_smoothing))
        # Final covariance matrix
        covariance_matrix = diagonal_covarianbce_matrix*smoothing_covariance_matrix*diagonal_covarianbce_matrix.H
        #normalized_covariance_matrix = covariance_matrix/np.max(covariance_matrix)
        if self.covariance_parameters["causal"] is "yes":
            return covariance_matrix
        else:
            return np.real(normalized_covariance_matrix)
    #
    def __get_total_covariance_matrix(self, covariance_matrix, observation_models, output_time_series_index):
        total_covariance_matrix = 0.
        number_frequencies = covariance_matrix.shape[0]
        index = 0
        for model in observation_models:
            if index != output_time_series_index:
               total_covariance_matrix += model*covariance_matrix*model.H
            index += 1
        total_covariance_matrix += np.matrix(np.identity(number_frequencies))*self.noise_level**2
        return total_covariance_matrix
    #
    def __get_dynamic_polynomials(self):
        if self.dynamic_type is "Relaxation":
            dynamic_polynomials = []
            for constant in self.dynamic_parameters["relaxation_constants"]:
                dynamic_polynomials += [constant - 1j*self.frequency_range]
            return dynamic_polynomials
        elif self.dynamic_type is "Oscillation":
            dynamic_polynomials = []
            print self.dynamic_parameters["number_sources"]
            for process_index in range(0,self.dynamic_parameters["number_sources"]):
                relaxation_constant = self.dynamic_parameters["relaxation_constants"][process_index]
                frequency = self.dynamic_parameters["frequency"][process_index] 
                dynamic_polynomials += [-self.frequency_range**2 - 1j*relaxation_constant*self.frequency_range + frequency**2]
            return dynamic_polynomials
        else:
            print "The requested dynamics is currently not implemented"
            raise
    #
    def __get_moving_average_kernels(self):
        moving_average_kernels = []
        for constant in self.dynamic_parameters["moving_average_time_constants"]:
           moving_average_kernels += [1. - 1./(2*np.pi*(constant - 1j*self.frequency_range))]
        return moving_average_kernels
    #
    def __get_frequency_range(self):
        max_frequency = (2*np.pi)/(2*self.time_parameters["time_step"])
        frequency_step = (2*np.pi)/(self.time_parameters["time_period"])
        self.frequency_range = np.arange(-max_frequency, max_frequency, frequency_step)
        return 
    #
    def __get_log_marginal_likelihood(self, covariance_matrix, data, time_series_index, jitter):
        log_marginal_likelihood = 0
        number_points = len(data[0][time_series_index,:])
        jittered_covariance_matrix = covariance_matrix + jitter*np.identity(number_points)
        cholesky_factor = np.linalg.cholesky(jittered_covariance_matrix)
        log_determinant = 2*np.sum(np.log(np.diag(cholesky_factor)))
        for sample in data:
            time_series = sample[time_series_index,:].T
            whitened_time_series = matrix_division(divider = cholesky_factor, 
                                                   divided = time_series, 
                                                   side = "left", 
                                                   cholesky = "no")
            data_factor = whitened_time_series.T*whitened_time_series
            sample_log_marginal_likelihood = - 0.5*number_points*np.log(np.pi) - 0.5*log_determinant - 0.5*data_factor
            log_marginal_likelihood += sample_log_marginal_likelihood
        return log_marginal_likelihood
    
    def __get_AR1_dynamic_covariance_matrix(self, relaxation, amplitude):
        time_difference_matrix = self.time_parameters["time_difference_matrix"]
        dynamic_covariance_matrix = (amplitude**2)*np.exp(-relaxation*np.abs(time_difference_matrix))
        return dynamic_covariance_matrix
    
    def __get_dynamic_parameters(self, dynamic_parameters_range, data, time_series_index, jitter):
        if self.dynamic_type is "Relaxation":
            # relaxation range
            rel_step = dynamic_parameters_range["relaxation_constant"]["step"]
            rel_min = dynamic_parameters_range["relaxation_constant"]["min"]
            rel_max =dynamic_parameters_range["relaxation_constant"]["max"]
            relaxation_range = np.arange(rel_min, rel_max, rel_step)
            # noise amplitude range
            amp_step = dynamic_parameters_range["amplitude"]["step"]
            amp_min = dynamic_parameters_range["amplitude"]["min"]
            amp_max = dynamic_parameters_range["amplitude"]["max"]
            amplitude_range = np.arange(amp_min, amp_max, amp_step)

            log_marginal_likelihood = np.zeros(shape = (len(relaxation_range),len(amplitude_range)))
            relaxation_index = -1
            for relaxation in relaxation_range:
                relaxation_index += 1
                amplitude_index = -1
                for amplitude in amplitude_range:
                    amplitude_index += 1
                    dynamic_covariance_matrix = self.__get_AR1_dynamic_covariance_matrix(relaxation, amplitude)
                    log_marginal_likelihood[relaxation_index,amplitude_index] = self.__get_log_marginal_likelihood(dynamic_covariance_matrix, data, time_series_index, jitter)
            flat_argmax_index = np.argmax(log_marginal_likelihood)
            argmax_index = np.unravel_index(flat_argmax_index, dims = np.shape(log_marginal_likelihood))
            self.simulator.dynamic_parameters["relaxation_constants"][time_series_index] = relaxation_range[argmax_index[0]]
            self.simulator.dynamic_parameters["amplitude"][time_series_index] = amplitude_range[argmax_index[1]]
            return 
        else:
            print "The method for learning the dynamic parameters is currently only implemented for relaxation dynamics"
            raise
    #
    def learn_dynamic_parameters(self, dynamic_parameters_range, jitter):
        if self.dynamic_type is "Relaxation":
            if len(self.dynamic_parameters) == 0:
                self.dynamic_parameters = self.simulator.dynamic_parameters
            number_sources = self.dynamic_parameters["number_sources"]
            self.dynamic_parameters["relaxation_constants"] = np.zeros(shape = (number_sources,1))
            self.dynamic_parameters["amplitude"] = np.zeros(shape = (number_sources,1))
            for time_series_index in range(0,number_sources):
                self.__get_dynamic_parameters(dynamic_parameters_range, 
                                              self.samples, 
                                              time_series_index, 
                                              jitter)
            return  
        else:
            print "The method for learning the dynamic parameters is currently only implemented for relaxation dynamics"
            raise
    #
    def __squared_exponential_covariance(self, time_lag, length_scale):
        covariance = np.exp(-np.power(time_lag,2)/(2*length_scale**2))/(np.sqrt(2*np.pi)*length_scale)
        complex_covariance = sig_tools.hilbert(covariance)
        return complex_covariance
    #
    def run_analysis(self):
        self.__get_frequency_range()
        dynamic_polynomials = self.__get_dynamic_polynomials()
        moving_average_kernels = self.__get_moving_average_kernels()
        covariance_matrix = self.__get_covariance_matrix()
        self.connectivity_kernels_list = []
        for sample in self.samples:
            fourier_time_series = self.__get_fourier_time_series(sample)
            modified_processes = self.__get_modified_processes(fourier_time_series, 
                                                               dynamic_polynomials)
            observation_models = self.__get_observation_models(fourier_time_series, 
                                                               moving_average_kernels)
            connectivity_kernels = []
            for output_time_series_index in range(0, self.dynamic_parameters["number_sources"]):
                connectivity_kernels_to_output = []
                total_covariance_matrix = self.__get_total_covariance_matrix(covariance_matrix, 
                                                                             observation_models,
                                                                             output_time_series_index)
                for input_time_series_index in range(0, self.dynamic_parameters["number_sources"]):
                    if output_time_series_index ==  input_time_series_index:
                        connectivity_kernels_to_output += [[]] # the model does not include self-connectivity
                    else:
                        connectivity_kernel = covariance_matrix*observation_models[input_time_series_index].H*matrix_division(divider = total_covariance_matrix,
                                                                                                                              divided = np.matrix(modified_processes[output_time_series_index]).T, 
                                                                                                                              side = "left", 
                                                                                                                              cholesky = "no")
                                                                                                                              
                        time_domain_connectivity_kernel = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(connectivity_kernel),axis = 0))
                        connectivity_kernels_to_output += [time_domain_connectivity_kernel]
                connectivity_kernels += [connectivity_kernels_to_output]
            self.connectivity_kernels_list += [connectivity_kernels]
        return
    #
    def plot_connectivity_kernel(self, time_range, sample_index, output_index, input_index):
        plt.plot(time_range, self.connectivity_kernels_list[sample_index][output_index][input_index])
        return
    #
    def plot_connectivity_statistics(self, time_range, output_index, input_index, error_bars):
        number_samples = len(self.connectivity_kernels_list)
        mean = self.connectivity_statistics["mean"][output_index][input_index]
        standard_deviation = np.sqrt(self.connectivity_statistics["variance"][output_index][input_index])
        if error_bars is "standard_error":
            lower_bound = mean - standard_deviation/np.sqrt(number_samples)
            upper_bound = mean + standard_deviation/np.sqrt(number_samples)
        elif error_bars is "standard_deviation":
            lower_bound = mean - standard_deviation
            upper_bound = mean + standard_deviation
        elif error_bars is "95_interval":
            lower_bound = mean - 1.96*standard_deviation/np.sqrt(number_samples)
            upper_bound = mean + 1.96*standard_deviation/np.sqrt(number_samples)
        else:
            print "the requested error bars are currently not implemented"
        fig, ax = plt.subplots(1)
        plt.plot(time_range, mean.flatten())
        ax.fill_between(time_range, lower_bound.flatten(), upper_bound.flatten(), facecolor='yellow', alpha=0.5)
        return
    #
    def __get_inputOutput_connectivity_list(self, output_index, input_index):
        inputOutput_connectivity_list = []
        for connectivity_kernels in self.connectivity_kernels_list:
            inputOutput_connectivity_list += [connectivity_kernels[output_index][input_index]]
        return inputOutput_connectivity_list
    #
    def __get_mean_connectivity(self, inputOutput_connectivity_list):
        mean_connectivity = 0.
        number_samples = len(inputOutput_connectivity_list)
        for connectivity_kernel in inputOutput_connectivity_list:
            mean_connectivity += connectivity_kernel/float(number_samples)
        return mean_connectivity
    #
    def __get_variance_connectivity(self, mean_connectivity, inputOutput_connectivity_list):
        variance_connectivity = -np.power(mean_connectivity,2)
        number_samples = len(inputOutput_connectivity_list)
        for connectivity_kernel in inputOutput_connectivity_list:
            variance_connectivity += np.power(connectivity_kernel,2)/float(number_samples)
        return variance_connectivity
    #
    def compute_connectivity_statistics(self):
        mean_connectivity = []
        variance_connectivity = []
        sd_connectivity = []
        self.connectivity_statistics = {"mean": mean_connectivity, 
                                        "variance": variance_connectivity}
        for output_index in range(0, self.dynamic_parameters["number_sources"]):
            mean_connectivitys_row = []
            variance_connectivity_row = []
            for input_index in range(0, self.dynamic_parameters["number_sources"]):
                if output_index != input_index:
                    inputOutput_connectivity_list = self.__get_inputOutput_connectivity_list(output_index, input_index)
                    mean = self.__get_mean_connectivity(inputOutput_connectivity_list)
                    variance = self.__get_variance_connectivity(mean, inputOutput_connectivity_list)
                    mean_connectivitys_row += [mean]
                    variance_connectivity_row += [variance]  
                else:
                    mean_connectivitys_row += [[]]
                    variance_connectivity_row += [[]]   
            mean_connectivity += [mean_connectivitys_row]
            variance_connectivity += [variance_connectivity_row]
        return
            


# In[ ]:



