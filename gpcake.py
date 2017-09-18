import numpy as np
import scipy.signal as sig_tools
import scipy.cluster as cluster
import scipy.spatial as spatial
import warnings
import sys
sys.setrecursionlimit(20000)

from utility import matrix_division
import utility 
import diagnostics

class gpcake(object):
    #
    def __init__(self):
        self.dynamic_type = []
        self.number_sources = []
        self.time_parameters = {}
        self.frequency_range = []
        self.covariance_parameters = {"time_scale": 0.4,
                                      "spectral_smoothing": 4*np.pi*2*10**-1,
                                      "causal": "yes"}
        self.dynamic_parameters = {}
        self.noise_level = 0.05
        self.connectivity_kernels_list = []
        self.connectivity_statistics = {"mean": [],
                                        "variance": []}
        self.bayesfactors = []
        self.causal_obs_model = None
        self.parallelthreads = 1
        self.structural_constraint = None
    #
    def initialize_time_parameters(self, time_step, time_period):
        self.time_parameters = {"time_period": time_period,
                                          "time_step": time_step}

        self.time_meshgrid = self.get_time_meshgrid(self.time_parameters["time_period"],
                                                                        self.time_parameters["time_step"])

        self.time_parameters["time_difference_matrix"] = self.time_meshgrid["time_difference_matrix"]
        self.__set_frequency_range()
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
    
    ###
    ### Parameter fitting using empirical Bayes
    ###
    
    def empirical_least_squares_analysis(self, time_domain_trials):
        ## private functions ##
        def get_fourier_tensor(time_domain_trials):
            fourier_tensor = np.zeros(shape = (number_trials, number_sources, number_frequencies), dtype = "complex")
            for trial_index, trial in enumerate(time_domain_trials):
                fourier_tensor[trial_index,:,:] = self.__get_fourier_time_series(trial)
            return fourier_tensor
        def get_feature_matrices(fourier_tensor, dynamic_polynomials):
            feature_matrices_list = []
            for source_index in range(0, number_sources):
                frequency_resolved_matrices = []
                for frequency_index, polynomial_value in enumerate(dynamic_polynomials[source_index]):
                    feature_matrix = np.matrix(fourier_tensor[:,np.arange(0,number_sources) != source_index,frequency_index]/polynomial_value)
                    frequency_resolved_matrices += [feature_matrix]
                feature_matrices_list += [frequency_resolved_matrices]
            return feature_matrices_list
        def get_least_squares_results(feature_matrices_list, fourier_tensor):
            ls_result_list = []
            for source_index, feature_matrices in enumerate(feature_matrices_list):
                frequency_resolved_ls_results = []
                for frequency_index, feature_matrix in enumerate(feature_matrices):
                    inverse_correlation = np.linalg.inv(feature_matrix.H*feature_matrix)
                    fourier_data = np.matrix(fourier_tensor[:,source_index,frequency_index]).T
                    ls_estimate = inverse_correlation*feature_matrix.H*fourier_data
                    predictions = feature_matrix*ls_estimate
                    residual_variance = np.real(np.sum(np.power(np.abs(fourier_data - predictions),2))/(number_trials - 2))
                    null_model_residual_variance = np.real(np.sum(np.power(np.abs(fourier_data),2))/(number_trials - 2))
                    log_likelihood_ratio = np.log(null_model_residual_variance) - np.log(residual_variance)
                    ls_covariance = inverse_correlation*float(residual_variance)
                    ls_second_moment = ls_covariance + ls_estimate*ls_estimate.H
                    frequency_resolved_ls_results += [{"ls_estimate": ls_estimate,
                                                       "corrected_residual_variance": np.abs(dynamic_polynomials[source_index][frequency_index])**2*residual_variance,
                                                       "ls_covariance": np.diagonal(ls_covariance),
                                                       "ls_second_moment": np.diagonal(ls_second_moment),
                                                       "log_likelihood_ratio": log_likelihood_ratio}]
                ls_result_list += [frequency_resolved_ls_results]
            return ls_result_list
        ##
        number_trials = len(time_domain_trials)
        number_sources, number_frequencies = time_domain_trials[0].shape
        dynamic_polynomials = self.time_parameters["time_step"]*np.array(self.__get_dynamic_polynomials())
        fourier_tensor = get_fourier_tensor(time_domain_trials)
        feature_matrices_list = get_feature_matrices(fourier_tensor, dynamic_polynomials)
        ls_results = get_least_squares_results(feature_matrices_list, fourier_tensor)
        return ls_results
    #
    def empirical_bayes_parameter_fit(self, time_domain_trials, learn_structural_constraint=True, show_diagnostics=False):
        ## private function ##
        def rearrange_results(least_squares_results, field):
            ##
            def correct_index(in_index,out_index):
                if in_index < out_index:
                    return in_index
                else:
                    return in_index - 1
            ##
            kernel_list = []
            for out_index in range(0, number_sources):
                kernel_list_row = []
                for in_index in range(0, number_sources):
                    if out_index == in_index:
                        kernel_array = []
                    else:
                        kernel = []
                        corr_in_index = correct_index(in_index, out_index)
                        for frequency_index in range(0, number_frequencies):
                            input_results = least_squares_results[out_index][frequency_index][field]
                            if type(input_results) is np.float64:
                                kernel += [np.real(input_results)]
                            else:
                                kernel += [np.real(input_results[corr_in_index])]
                        kernel_array = np.array(kernel)
                    kernel_list_row += [kernel_array]
                kernel_list += [kernel_list_row]  
            return kernel_list 
        #
        def get_attribute_matrices(result_matrices):
            ##
            def fit_gaussian(empirical_kernel, 
                         grid_size = 6000):
                ## private lambdas ##
                unzip              = lambda lst: zip(*lst)
                deviation_function = lambda x,y: np.sum(np.abs(x - y))
                normalize          = lambda x: x/np.sum(x)
                smoothing_function = lambda x,l: np.exp(-np.power(x,2)/(2*l**2))/(np.sqrt(np.pi*2*l**2))
                ## function body ##
                if len(empirical_kernel) == 0:
                    return []
                else:
                    frequencies, values = unzip([(freq, val) 
                                                 for (freq,val) in zip(self.frequency_range, empirical_kernel) 
                                                 if frequency_filter(freq, freq_bound)])
                    scale_range = np.linspace(10**-4,freq_bound,grid_size)
                    deviations = [deviation_function(normalize(values), smoothing_function(frequencies,l))
                                  for l in scale_range]
                    optimal_scale = scale_range[np.argmin(deviations)]
                    return optimal_scale
            def get_scalar_matrix(kernel_matrix):
                reduce_kernel = lambda kernel: [val 
                                                for (freq, val) in zip(self.frequency_range,kernel) 
                                                if frequency_filter(freq,freq_bound/3)]
                get_scalar = lambda kernel: np.sum(reduce_kernel(kernel))
                scalar_matrix = utility.nested_map(get_scalar, kernel_matrix)
                return scalar_matrix
            symmetrize = lambda A: A + np.transpose(A)
            ##
            strength_matrix = np.array(get_scalar_matrix(result_matrices["ls_second_moment"]))
            np.fill_diagonal(strength_matrix, 0.5)
            
            residual_matrix = np.array(get_scalar_matrix(result_matrices["corrected_residual_variance"]))
            np.fill_diagonal(residual_matrix, 1)
            
            empirical_ls_estimate_matrix = np.array(get_scalar_matrix(utility.nested_map(lambda x: np.power(np.abs(x),2),
                                                                                         result_matrices["ls_estimate"],
                                                                                         ignore_diagonal=True)))
            np.fill_diagonal(empirical_ls_estimate_matrix, 0)
            effect_size_matrix = np.divide(empirical_ls_estimate_matrix,residual_matrix)
            
            log_likelihood_matrix = np.array(get_scalar_matrix(result_matrices["log_likelihood_ratio"]))
            np.fill_diagonal(log_likelihood_matrix, 0.)
            
            scale_matrix = np.array(utility.nested_map(fit_gaussian, 
                                                       result_matrices["ls_second_moment"], 
                                                       ignore_diagonal=True))
            np.fill_diagonal(scale_matrix, 0)     
            
            if show_diagnostics:
                diagnostics.plot_scale_fit(result_matrices["ls_second_moment"], 
                                           scale_matrix, 
                                           self.frequency_range, 
                                           freq_bound                                           
                                           )
            
            symmetric_scale_matrix = (symmetrize(strength_matrix*scale_matrix)/(symmetrize(strength_matrix)))
            
            return {"scale":                  symmetric_scale_matrix.tolist(), 
                    "connectivity_strength":  strength_matrix.tolist(),
                    "log_likelihood_ratio":   log_likelihood_matrix.tolist(),
                    "effect_size":            effect_size_matrix.tolist(),
                    "residuals":              residual_matrix.tolist()
                   }
        #
        def get_attribute_centroid_matrix(dic_attribute_matrices):
            ## private functions ##
            def replace_with_centroid(centroids, 
                                      attributes_std, 
                                      list_attribute_matrices):
                rescale = lambda x: np.divide(x,attributes_std)
                distance = lambda x,y: np.linalg.norm(np.array(x)-np.array(y))
                closest_centroid = lambda x: np.multiply(centroids[np.argmin([distance(rescale(x),c) 
                                                                   for c in centroids]),:],attributes_std)
                quantized_matrix = [map(closest_centroid, row)
                                    for row in utility.nested_zip(*list_attribute_matrices)]
                quantized_matrix = utility.fill_diagonal(quantized_matrix, [])
                return quantized_matrix
            flat_list = lambda lst: [item 
                                     for index1, sublist in enumerate(lst) 
                                     for index2, item in enumerate(sublist)
                                     if index1 != index2]
            def standardize(data_array):
                standardized_array = np.zeros(data_array.shape)
                attributes_std = np.array([np.std(row) for row in data_array]) 
                for attribute_index, scale_factor in enumerate(attributes_std):
                    standardized_array[attribute_index,:] = data_array[attribute_index,:] / scale_factor
                return (standardized_array, attributes_std)
            #
            ## method body ##
            list_attribute_keys = dic_attribute_matrices.keys()
            list_attribute_matrices = dic_attribute_matrices.values()
            flat_matrices = map(flat_list, list_attribute_matrices)
            standardized_data, attributes_std = standardize(np.array(flat_matrices))
            standardized_centroids,_ = cluster.vq.kmeans(np.transpose(standardized_data), 2)
            attribute_centroid_matrix = replace_with_centroid(standardized_centroids, 
                                                              attributes_std,
                                                              list_attribute_matrices)
            if show_diagnostics:
                diagnostics.plot_distances(standardized_data, standardized_centroids)                
            return (attribute_centroid_matrix, list_attribute_keys)
        #
        def get_structural_connectivity_matrix(attribute_centroid_matrix, list_attribute_keys):
            ## private functions ##
            def classify_edge(x):
                criteria_list = [criterion(x) for criterion in criteria]
                if all(criteria_list):
                    return 1
                elif any(criteria_list):
                    warnings.warn(message = "The connectivity criterion was not decisive")
                    return 1
                else:
                    return 0
            def get_nested_minimum(index, attribute_centroid_matrix):                
                minimum_value = float("inf")
                filled_matrix = utility.fill_diagonal(attribute_centroid_matrix, tuple_size*[minimum_value])
                return utility.nested_foldRight(lambda X,y: min(X[index], y),
                                                min,
                                                minimum_value,
                                                filled_matrix)                
            ##
            tuple_size = len(list_attribute_keys)
            strength_index = list_attribute_keys.index("connectivity_strength") 
            minimum_strength = get_nested_minimum(strength_index, attribute_centroid_matrix)
            strength_criterion = lambda x: x[strength_index]>minimum_strength
            
            effect_size_index = list_attribute_keys.index("effect_size")
            minimum_effect_size = get_nested_minimum(effect_size_index, attribute_centroid_matrix)
            effect_size_criterion = lambda x: x[effect_size_index]>minimum_effect_size
            
            log_lk_index = list_attribute_keys.index("log_likelihood_ratio")
            minimum_log_lk = get_nested_minimum(log_lk_index, attribute_centroid_matrix)
            log_lk_criterion = lambda x: x[log_lk_index]>minimum_log_lk
            
            criteria = [strength_criterion, effect_size_criterion, log_lk_criterion]
            filled_matrix = utility.fill_diagonal(attribute_centroid_matrix, tuple_size*[-float("inf")])
            structural_connectivity_matrix = utility.nested_map(classify_edge, filled_matrix, ignore_diagonal=False)            
            return structural_connectivity_matrix
        #
        def filter_attribute(attribute_centroid_matrix, 
                             list_attribute_keys, 
                             attribute):            
            attr_index = list_attribute_keys.index(attribute)
            attr_matrix = utility.nested_map(lambda L: L[attr_index],
                                             attribute_centroid_matrix,
                                             ignore_diagonal = True)
            return attr_matrix
        #
        def get_time_scale(attribute_centroid_matrix, 
                           list_attribute_keys):   
            spectral_scales = filter_attribute(attribute_centroid_matrix, 
                                               list_attribute_keys, 
                                               'scale')
            temporal_scales = utility.nested_map(lambda x: 1./x,
                                                  spectral_scales, 
                                                  ignore_diagonal = True)
            return temporal_scales
        #
        def get_noise_level(attribute_centroid_matrix, 
                           list_attribute_keys):  
            noise_levels = filter_attribute(attribute_centroid_matrix, 
                                            list_attribute_keys, 
                                            'residuals') 
            return noise_levels
        #            
        frequency_filter = lambda freq, freq_bound: ((freq > -freq_bound) & (freq < freq_bound))
        ## function body ##
        print 'Training GP CaKe parameters with empirical Bayes.' 
        number_sources, number_frequencies = time_domain_trials[0].shape 
        freq_bound = np.max(np.abs(self.frequency_range)) / 5.
        ##
        ls_results = self.empirical_least_squares_analysis(time_domain_trials)
        fields_list = ["ls_second_moment", "log_likelihood_ratio", "corrected_residual_variance", "ls_estimate"]
        result_matrices = {field: rearrange_results(ls_results, field)
                           for field in fields_list}
        
        dic_attribute_matrices = get_attribute_matrices(result_matrices)
        attribute_centroid_matrix, list_attribute_keys = get_attribute_centroid_matrix(dic_attribute_matrices)
        
        time_scale = get_time_scale(attribute_centroid_matrix, 
                                    list_attribute_keys)
        
        scale2shift_proportion = 1.        
        time_shift = utility.nested_map(lambda x: scale2shift_proportion*x, time_scale, ignore_diagonal=True)
        
        noise_level = get_noise_level(attribute_centroid_matrix, 
                                      list_attribute_keys)        
        noise_level = utility.fill_diagonal(noise_level, float('nan'))
        noise_level_vector = map(lambda noise_list: np.nanmean(noise_list), noise_level)
        
        scale2spectral_proportion = 100.
        spectral_smoothing = utility.nested_map(lambda x: scale2spectral_proportion*x, time_scale, ignore_diagonal=True)
        
        parameter_matrices = utility.nested_zip(time_scale, 
                                                time_shift, 
                                                spectral_smoothing)
        
        self.parameter_matrices = parameter_matrices
        self.noise_vector = noise_level_vector
        
        if learn_structural_constraint:
            structural_connectivity_matrix = get_structural_connectivity_matrix(attribute_centroid_matrix, list_attribute_keys)   
            self.structural_constraint = np.array(structural_connectivity_matrix)
            print 'Connectivity constraint: enabled.'
        else:
            print 'Connectivity constraint: disabled.'
                
        
        print 'Empirical Bayes procedure complete.'

    ###
    ### End of empirical Bayes parameter fitting
    ###
    
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
    def __remove_modified_process(self, Px, dynamic_polynomials):
        x = []
        index = 0
        for series in Px:
            x += [np.divide(Px[index] / dynamic_polynomials[index])]
            index += 1
        return x
    #
    #def __get_covariance_matrix(self, time_scale=None, spectral_smoothing=None):
    #    if time_scale is None:
    #        time_scale = self.covariance_parameters["time_scale"]
    #    if spectral_smoothing is None:
    #        spectral_smoothing = self.covariance_parameters["spectral_smoothing"]
    #    time_shift = self.covariance_parameters["time_shift"]        
    #    # Frequency grid
    #    freq_grid_x, freq_grid_y = np.meshgrid(self.frequency_range, self.frequency_range)
    #    # Stationary part
    #    stationary_covariance_function = lambda f,l,tau: np.exp(-1j*tau*f-f**2/(2*l**2))
    #    spectral_width = 2*np.pi*(1/time_scale)
    #    diagonal_covariance_matrix = np.matrix(np.diag(stationary_covariance_function(self.frequency_range, spectral_width, time_shift)))
    #    # Nonstationary part
    #    smoothing_covariance_matrix = np.matrix(self.__squared_exponential_covariance(freq_grid_x - freq_grid_y, spectral_smoothing))
    #    # Final covariance matrix
    #    covariance_matrix = diagonal_covariance_matrix*smoothing_covariance_matrix*diagonal_covariance_matrix.H
    #    #
    #    if self.covariance_parameters["causal"] is "yes":
    #        return covariance_matrix
    #    else:
    #        return np.real(covariance_matrix)
        
    def __get_covariance_matrices(self):     
        ## private functions
        def get_covariance_matrix(parameter_tuple):
            stationary_covariance_function = lambda f,l,tau: np.exp(-1j*tau*f-f**2/(2*l**2))        
            time_scale = parameter_tuple[0]
            time_shift = parameter_tuple[1]
            spectral_smoothing = parameter_tuple[2]
            spectral_width = 2*np.pi*(1/time_scale)        
            diagonal_covariance_matrix = np.matrix(np.diag(stationary_covariance_function(self.frequency_range, 
                                                                                          spectral_width, 
                                                                                          time_shift)))
            # Nonstationary part
            smoothing_covariance_matrix = np.matrix(self.__squared_exponential_covariance(freq_grid_x - freq_grid_y, 
                                                                                          spectral_smoothing))
            # Final covariance matrix
            covariance_matrix = diagonal_covariance_matrix*smoothing_covariance_matrix*diagonal_covariance_matrix.H
            return covariance_matrix
        # Frequency grid
        freq_grid_x, freq_grid_y = np.meshgrid(self.frequency_range, self.frequency_range)        
        covariance_matrices = utility.nested_map(get_covariance_matrix, self.parameter_matrices, ignore_diagonal=True)
        return covariance_matrices
    #
    def __get_total_covariance_matrix(self, covariance_matrices, observation_models, output_time_series_index, noise_level):
        """
        K_f = [\sum_i Gamma_i K Gamma^H_i + sigma^2 I]
        """
        if noise_level == None:
            noise_level = self.noise_level            
        total_covariance_matrix = 0.
        number_frequencies = len(covariance_matrices[0][1])
        for input_index, model in enumerate(observation_models): 
            if input_index != output_time_series_index:
                covariance_matrix = covariance_matrices[output_time_series_index][input_index]
                total_covariance_matrix += model*covariance_matrix*model.H
        total_covariance_matrix += np.matrix(np.identity(number_frequencies)) * np.power(noise_level, 2)
        return total_covariance_matrix
    #
    def __get_dynamic_polynomials(self):
        frequency_range = self.__get_frequency_range()
        if self.dynamic_type is "Relaxation":
            dynamic_polynomials = []
            for constant in self.dynamic_parameters["relaxation_constants"]:
                dynamic_polynomials += [constant - 1j*frequency_range]
            return dynamic_polynomials
        elif self.dynamic_type is "Oscillation":
            dynamic_polynomials = []
            for process_index in range(0,self.dynamic_parameters["number_sources"]):
                relaxation_constant = self.dynamic_parameters["relaxation_constants"][process_index]
                frequency = self.dynamic_parameters["frequency"][process_index]
                dynamic_polynomials += [-self.frequency_range**2 - 1j*relaxation_constant*frequency_range + frequency**2]
            return dynamic_polynomials
        else:
            print("The requested dynamics is currently not implemented")
            raise
    #
    def __get_moving_average_kernels(self):
        moving_average_kernels = []
        for constant in self.dynamic_parameters["moving_average_time_constants"]:
            moving_average_kernels += [1. - 1./(2*np.pi*(constant - 1j*self.frequency_range))]
        return moving_average_kernels
    #
    def __set_frequency_range(self):
        max_frequency = (2*np.pi)/(2*self.time_parameters["time_step"])
        frequency_step = (2*np.pi)/(self.time_parameters["time_period"])
        self.frequency_range = np.arange(-max_frequency, max_frequency, frequency_step)
    #
    def __get_frequency_range(self):
        if len(self.frequency_range)==0:
            self.__set_frequency_range()        
        return self.frequency_range
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
    def __get_dynamic_parameters(self, dynamic_parameters_range, data, time_series_index, jitter):
        if self.dynamic_type is "Relaxation":
            # relaxation range
            rel_step = dynamic_parameters_range["relaxation_constant"]["step"]
            rel_min = dynamic_parameters_range["relaxation_constant"]["min"]
            rel_max = dynamic_parameters_range["relaxation_constant"]["max"]
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
            self.dynamic_parameters["relaxation_constants"][time_series_index] = relaxation_range[argmax_index[0]]
            self.dynamic_parameters["amplitude"][time_series_index] = amplitude_range[argmax_index[1]]
            return
        else:
            print("The method for learning the dynamic parameters is currently only implemented for relaxation dynamics")
            raise
            
    #
    def learn_dynamic_parameters(self, data, dynamic_parameters_range):
        jitter = 10 ** -15 # numerical stability
        if self.dynamic_type is "Relaxation":
            number_sources = self.dynamic_parameters["number_sources"]
            self.dynamic_parameters = {}
            self.dynamic_parameters["relaxation_constants"] = np.zeros(shape = (number_sources,1))
            self.dynamic_parameters["amplitude"] = np.zeros(shape = (number_sources,1))
            for time_series_index in range(0,number_sources):
                self.__get_dynamic_parameters(dynamic_parameters_range,
                                              data,
                                              time_series_index,
                                              jitter)
            self.dynamic_parameters["moving_average_time_constants"] = 1e15 * np.ones(shape = (number_sources,1))
            self.dynamic_parameters["number_sources"] = number_sources
            return
        else:
            print("The method for learning the dynamic parameters is currently only implemented for relaxation dynamics")
            raise
    #
    def __posterior_kernel_cij(self, obs_model, cov_matrix, total_cov_matrix, dynamic_modified_fourier_series):
        """
        The bread-and-butter of the method.
        
        """      
        y = np.matrix(dynamic_modified_fourier_series).T
        c_ij = cov_matrix * obs_model.H  * matrix_division(divider = total_cov_matrix, divided = y, side = "left", cholesky = "no")
            
        return np.real(np.array(c_ij).flatten())
    #
    def __posterior_kernel_cij_temporal(self, obs_model, cov_matrix, total_cov_matrix, dynamic_modified_fourier_series):
        """
        The bread-and-butter of the method.        
        """        
        y = np.matrix(dynamic_modified_fourier_series).T
        c_ij = cov_matrix * obs_model.H  * matrix_division(divider = total_cov_matrix, divided = y, side = "left", cholesky = "no")
            
        return np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.array(c_ij).flatten()), axis = 0)))
    #
    def __squared_exponential_covariance(self, time_lag, length_scale):
        covariance = np.exp(-np.power(time_lag,2)/(2*length_scale**2))/(np.sqrt(2*np.pi)*length_scale)
        complex_covariance = sig_tools.hilbert(covariance)
        return complex_covariance
    #
    def run_analysis(self, data, onlyTrials=False, show_diagnostics=False):
        ## private functions
        def run_analysis_body(sample):        
            #(nsources, nfrequencies) = np.shape(sample)        
            sample_connectivity = np.zeros((nsources, nsources, nfrequencies))

            x = self.__get_fourier_time_series(sample)
            Px = self.__get_modified_processes(x, dynamic_polynomials)
            observation_models = self.__get_observation_models(x, moving_average_kernels)
            for j in range(0, nsources): # target
                total_covariance_matrix = self.__get_total_covariance_matrix(covariance_matrices, # pxp matrices of temporal covariances
                                                                             observation_models, 
                                                                             j, 
                                                                             self.noise_vector[j]) # check this!
                for i in range(0, nsources): # source
                    if self.structural_constraint[i,j]:
                        connectivity_kernel = self.__posterior_kernel_cij_temporal(observation_models[i], 
                                                                                   covariance_matrices[i][j], # check this! [i,j] or [j,i]??
                                                                                   total_covariance_matrix, 
                                                                                   Px[j])

                        sample_connectivity[i, j, :] = connectivity_kernel
            return sample_connectivity
        #
        def run_analysis_serial(data):            
            connectivity = np.zeros((nsamples, nsources, nsources, nfrequencies))
            for s_ix, sample in enumerate(data):              
                connectivity[s_ix, :, :, :] = run_analysis_body(sample)
            return connectivity
        #
        def run_analysis_parallel(data):
            ## private function
            def run_analysis_parallel_wrapper(self, parallel_args_struct):
                return self.__run_analysis_body(sample=parallel_args_struct['sample'], 
                                                moving_average_kernels=parallel_args_struct['MA'], 
                                                covariance_matrix=parallel_args_struct['cov'], 
                                                dynamic_polynomials=parallel_args_struct['dypoly'])
            # function body
            connectivity = np.zeros((nsamples, nsources, nsources, nfrequencies))

            # Initialize parallelization
            from multiprocessing.dummy import Pool as ThreadPool 
            pool = ThreadPool(processes=self.parallelthreads)       

            parallel_args = []      

            for sample in data: #?
                parallel_args_struct = {}
                parallel_args_struct['sample'] = sample
                parallel_args_struct['MA'] = moving_average_kernels
                parallel_args_struct['cov'] = covariance_matrices
                parallel_args_struct['dypoly'] = dynamic_polynomials
                parallel_args += [parallel_args_struct]

            # Execute parallel computation
            parallel_results_list = pool.map(run_analysis_parallel_wrapper, parallel_args)    

            # Collect results
            for i in range(0, nsamples):
                connectivity[i,:,:,:] = parallel_results_list[i]

            pool.close()
            pool.join()
            return connectivity
        #
        def run_analysis_parallel_flatloop(data):
            ## private function
            def posterior_kernel_cij_temporal_parwrap(parallel_args_struct):
                """
                The bread-and-butter of the method.

                """     
                # unpack
                y = np.matrix(parallel_args_struct['Px_j']).T
                covariance_matrices = parallel_args_struct['cov']     
                observation_models = parallel_args_struct['obs_models']
                i = parallel_args_struct['i']
                j = parallel_args_struct['j']
                total_cov_matrix = self.__get_total_covariance_matrix(covariance_matrices, 
                                                                      observation_models, 
                                                                      j, 
                                                                      self.noise_level)   

                c_ij = covariance_matrices * observation_models[i].H  \
                       * matrix_division(divider = total_cov_matrix, divided = y, side = "left", cholesky = "no")

                return np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.array(c_ij).flatten()), axis = 0)))
            # body
            connectivity = np.zeros((nsamples, nsources, nsources, nfrequencies))


            # Initialize parallelization
            from multiprocessing.dummy import Pool as ThreadPool 
            pool = ThreadPool(processes=self.parallelthreads)    

            parallel_iterable = []      

            for k, sample in enumerate(data): #?
                x = self.__get_fourier_time_series(sample)
                Px = self.__get_modified_processes(x, dynamic_polynomials)
                observation_models = self.__get_observation_models(x, moving_average_kernels)            
                for j in range(0, nsources):               
                    for i in range(0, nsources):
                        if self.structural_constraint[i,j]:
                            parallel_args_struct = {}
                            parallel_args_struct['cov'] = covariance_matrices[i]
                            parallel_args_struct['k'] = k
                            parallel_args_struct['obs_models'] = observation_models
                            parallel_args_struct['i'] = i
                            parallel_args_struct['j'] = j
                            parallel_args_struct['Px_j'] = Px[j]
                            parallel_iterable += [parallel_args_struct]

            # Execute parallel computation
            parallel_results_list = pool.map(posterior_kernel_cij_temporal_parwrap, parallel_iterable)  
            # todo: properly unwrap this, instead of this ugly hack:
            for m, result in enumerate(parallel_results_list):            
                k = parallel_iterable[m]['k']
                i = parallel_iterable[m]['i']
                j = parallel_iterable[m]['j']
                connectivity[k,i,j,:] = result
            return connectivity
        ## function body of run_analysis()
        
        dynamic_polynomials = self.__get_dynamic_polynomials()
        moving_average_kernels = self.__get_moving_average_kernels()
        covariance_matrices = self.__get_covariance_matrices()
        (nsamples, nsources, nfrequencies) = np.shape(np.asarray(data))
        self.__get_frequency_range()
        
        if self.structural_constraint is None:
            self.structural_constraint = np.ones(shape=(nsources, nsources)) - np.eye(nsources)   
            
        if show_diagnostics:  
            print 'GP CaKe parameters:'
            print '\nTime scales (nu_ij):'
            print np.array(utility.fill_diagonal(utility.nested_map(lambda x: x[0], self.parameter_matrices), 0))
            print '\nTime shifts (t_ij):'
            print np.array(utility.fill_diagonal(utility.nested_map(lambda x: x[1], self.parameter_matrices), 0))
            print '\nSpectral smoothing: (theta_ij)' 
            print np.array(utility.fill_diagonal(utility.nested_map(lambda x: x[2], self.parameter_matrices), 0))
            print '\nNoise levels (sigma_i):'
            print np.array(self.noise_vector)
            print '\nConnectivity constraint (G_ij):'
            print self.structural_constraint
            
            utility.tic()
        
        connectivity = None
        if self.parallelthreads > 1 and not onlyTrials:
            print('Parallel implementation (p = {:d}).'.format(self.parallelthreads))            
            connectivity = run_analysis_parallel_flatloop(data)
        elif self.parallelthreads > 1 and onlyTrials:
            print('Parallel implementation (p = {:d}, over trials only).'.format(self.parallelthreads))            
            connectivity = run_analysis_parallel(data)
        else:      
            #print('Serial implementation.')
            connectivity = run_analysis_serial(data)
        
        if show_diagnostics:
            utility.toc()
        return connectivity
    #
    def get_statistic(self, stat, output_index, input_index):
        stat = self.connectivity_statistics[stat][output_index][input_index]
        return stat.flatten()
    #        
