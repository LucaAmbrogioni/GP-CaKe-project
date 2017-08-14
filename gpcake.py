import numpy as np
import scipy.signal as sig_tools
from utility import matrix_division



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
    #
    def initialize_time_parameters(self, time_step, time_period):
        self.time_parameters = {"time_period": time_period,
                                          "time_step": time_step}

        self.time_meshgrid = self.get_time_meshgrid(self.time_parameters["time_period"],
                                                                        self.time_parameters["time_step"])

        self.time_parameters["time_difference_matrix"] = self.time_meshgrid["time_difference_matrix"]
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
    def __remove_modified_process(self, Px, dynamic_polynomials):
        x = []
        index = 0
        for series in Px:
            x += [np.divide(Px[index] / dynamic_polynomials[index])]
            index += 1
        return x
    #
    def __get_covariance_matrix(self, time_scale=None, spectral_smoothing=None):
        if time_scale is None:
            time_scale = self.covariance_parameters["time_scale"]
        if spectral_smoothing is None:
            spectral_smoothing = self.covariance_parameters["spectral_smoothing"]
        time_shift = self.covariance_parameters["time_shift"]

        # Frequency grid
        freq_grid_x, freq_grid_y = np.meshgrid(self.frequency_range, self.frequency_range)
        # Stationary part
        stationary_covariance_function = lambda f,l,tau: np.exp(-1j*tau*f-f**2/(2*l**2))
        spectral_width = 2*np.pi*(1/time_scale)
        diagonal_covariance_matrix = np.matrix(np.diag(stationary_covariance_function(self.frequency_range, spectral_width, time_shift)))
        # Nonstationary part
        smoothing_covariance_matrix = np.matrix(self.__squared_exponential_covariance(freq_grid_x - freq_grid_y, spectral_smoothing))
        # Final covariance matrix
        covariance_matrix = diagonal_covariance_matrix*smoothing_covariance_matrix*diagonal_covariance_matrix.H
        #
        if self.covariance_parameters["causal"] is "yes":
            return covariance_matrix
        else:
            return np.real(covariance_matrix)

    #
    def __get_total_covariance_matrix(self, covariance_matrix, observation_models, output_time_series_index, noise_level):
        """
        K_f = [\sum_i Gamma_i K Gamma^H_i + sigma^2 I]
        """
        if noise_level == None:
            noise_level = self.noise_level#            
        total_covariance_matrix = 0.
        number_frequencies = covariance_matrix.shape[0]
        index = 0
        for model in observation_models: 
            if index != output_time_series_index:
               total_covariance_matrix += model*covariance_matrix*model.H
            index += 1
        total_covariance_matrix += np.matrix(np.identity(number_frequencies))* np.power(noise_level,2)
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
            for process_index in range(0,self.dynamic_parameters["number_sources"]):
                relaxation_constant = self.dynamic_parameters["relaxation_constants"][process_index]
                frequency = self.dynamic_parameters["frequency"][process_index]
                dynamic_polynomials += [-self.frequency_range**2 - 1j*relaxation_constant*self.frequency_range + frequency**2]
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

    #
    def __squared_exponential_covariance(self, time_lag, length_scale):
        covariance = np.exp(-np.power(time_lag,2)/(2*length_scale**2))/(np.sqrt(2*np.pi)*length_scale)
        complex_covariance = sig_tools.hilbert(covariance)
        return complex_covariance
    #
    
    def __run_analysis_body(self, sample, moving_average_kernels, covariance_matrix, dynamic_polynomials):        
        (nsources, nfrequencies) = np.shape(sample)        
        sample_connectivity = np.zeros((nsources, nsources, nfrequencies))
                
        x = self.__get_fourier_time_series(sample)
        Px = self.__get_modified_processes(x, dynamic_polynomials)
        observation_models = self.__get_observation_models(x, moving_average_kernels)
        for j in range(0, nsources): # target
            total_covariance_matrix = self.__get_total_covariance_matrix(covariance_matrix, observation_models, j, self.noise_level)
            
            for i in range(0, nsources): # source
                if j != i:
                    connectivity_kernel = self.__posterior_kernel_cij_temporal(observation_models[i], 
                                                                               covariance_matrix, 
                                                                               total_covariance_matrix, 
                                                                               Px[j])
                    
                    sample_connectivity[i, j, :] = connectivity_kernel
        return sample_connectivity
    #  
    def __run_analysis_serial(self, data):
        dynamic_polynomials = self.__get_dynamic_polynomials()
    
        moving_average_kernels = self.__get_moving_average_kernels()
        covariance_matrix = self.__get_covariance_matrix()
        (nsamples, nsources, nfrequencies) = np.shape(np.asarray(data))
        connectivity = np.zeros((nsamples, nsources, nsources, nfrequencies))

        s_ix = 0
        for sample in data:              
            connectivity[s_ix, :, :, :] = self.__run_analysis_body(sample, moving_average_kernels, covariance_matrix, dynamic_polynomials)
            s_ix += 1
        return connectivity
    #
    def run_analysis(self, data, onlyTrials=False):
        self.__get_frequency_range()
        if self.parallelthreads > 1 and not onlyTrials:
            print('Parallel implementation (p = {:d}).'.format(self.parallelthreads))            
            return self.__run_analysis_parallel_flatloop(data)
        elif self.parallelthreads > 1 and onlyTrials:
            print('Parallel implementation (p = {:d}, over trials only).'.format(self.parallelthreads))            
            return self.__run_analysis_parallel(data)
        else:      
            print('Serial implementation.')
            return self.__run_analysis_serial(data)
            
    #    
    def __run_analysis_parallel_wrapper(self, parallel_args_struct):
        return self.__run_analysis_body(sample=parallel_args_struct['sample'], 
                                        moving_average_kernels=parallel_args_struct['MA'], 
                                        covariance_matrix=parallel_args_struct['cov'], 
                                        dynamic_polynomials=parallel_args_struct['dypoly'])
    
    #
    def __run_analysis_parallel(self, data):
        
        dynamic_polynomials = self.__get_dynamic_polynomials()

        moving_average_kernels = self.__get_moving_average_kernels()
        covariance_matrix = self.__get_covariance_matrix()
        (nsamples, nsources, nfrequencies) = np.shape(np.asarray(data))
        connectivity = np.zeros((nsamples, nsources, nsources, nfrequencies))
        
        # Initialize parallelization
        from multiprocessing.dummy import Pool as ThreadPool 
        pool = ThreadPool(processes=self.parallelthreads)       
        
        parallel_args = []      
                
        for sample in data: #?
            parallel_args_struct = {}
            parallel_args_struct['sample'] = sample
            parallel_args_struct['MA'] = moving_average_kernels
            parallel_args_struct['cov'] = covariance_matrix
            parallel_args_struct['dypoly'] = dynamic_polynomials
            parallel_args += [parallel_args_struct]
        
        # Execute parallel computation
        parallel_results_list = pool.map(self.__run_analysis_parallel_wrapper, parallel_args)    
        
        # Collect results
        for i in range(0, nsamples):
            connectivity[i,:,:,:] = parallel_results_list[i]
        
        pool.close()
        pool.join()
        return connectivity
    #
    def __posterior_kernel_cij_temporal_parwrap(self, parallel_args_struct):
        """
        The bread-and-butter of the method.
        
        """       
            
        # unpack
        y = np.matrix(parallel_args_struct['Px_j']).T
        covariance_matrix = parallel_args_struct['cov']     
        observation_models = parallel_args_struct['obs_models']
        i = parallel_args_struct['i']
        j = parallel_args_struct['j']
        total_cov_matrix = self.__get_total_covariance_matrix(covariance_matrix, observation_models, j, self.noise_level)   
        
        c_ij = covariance_matrix * observation_models[i].H  * matrix_division(divider = total_cov_matrix, divided = y, side = "left", cholesky = "no")
            
        return np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.array(c_ij).flatten()), axis = 0)))
    #
    def __run_analysis_parallel_flatloop(self, data):
        dynamic_polynomials = self.__get_dynamic_polynomials()

        moving_average_kernels  = self.__get_moving_average_kernels()
        covariance_matrix = self.__get_covariance_matrix()
        (nsamples, nsources, nfrequencies) = np.shape(np.asarray(data))
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
                    if j != i:
                        parallel_args_struct = {}
                        parallel_args_struct['cov'] = covariance_matrix
                        parallel_args_struct['k'] = k
                        parallel_args_struct['obs_models'] = observation_models
                        parallel_args_struct['i'] = i
                        parallel_args_struct['j'] = j
                        parallel_args_struct['Px_j'] = Px[j]
                        parallel_iterable += [parallel_args_struct]
        
        # Execute parallel computation
        parallel_results_list = pool.map(self.__posterior_kernel_cij_temporal_parwrap, parallel_iterable)  
        # todo: properly unwrap this, instead of this ugly hack:
        for m, result in enumerate(parallel_results_list):            
            k = parallel_iterable[m]['k']
            i = parallel_iterable[m]['i']
            j = parallel_iterable[m]['j']
            connectivity[k,i,j,:] = result
        return connectivity
        
    #
    
    def get_statistic(self, stat, output_index, input_index):
        stat = self.connectivity_statistics[stat][output_index][input_index]
        return stat.flatten()
    #        





