import numpy as np
import scipy as sp
from utility import matrix_division

class integroDifferential_simulator(object):
    #
    def __init__(self):
        self.dynamic_type = "Relaxation"
        self.dynamic_parameters = {"number_sources": 2,
                                   "connectivity_weights": np.array([[0, 1],[0, 0]]),
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
        block_green_matrix = sp.linalg.block_diag(*green_matrices) # creates block matrix from arrays
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
                                                   self.time_meshgrid) # needed for Hilbert transform in causal kernel
        kernel_matrices = self.__get_kernel_matrices(self.dynamic_parameters["connectivity_relaxations_constants"],
                                                     self.dynamic_parameters["connectivity_weights"],
                                                     self.time_meshgrid) # temporal locality
        moving_average_matrices = self.__get_moving_average_matrices(self.dynamic_parameters["moving_average_time_constants"],
                                                                     self.time_meshgrid) # assumption: smoothness kernel
        block_matrices = self.__get_block_matrices(green_matrices,
                                                   kernel_matrices,
                                                   moving_average_matrices)
        self.samples = []
        for sample_index in range(0, number_samples):
            self.samples += [self.__get_sample(block_matrices,
                                               self.dynamic_parameters["number_sources"],
                                               self.time_meshgrid["number_time_points"])]
    
    def run_spiking_network_sampler(self, number_samples):
        self.time_meshgrid = self.get_time_meshgrid(self.time_parameters["time_period"],
                                                    self.time_parameters["time_step"])

        self.action_potentials    = []
        self.membrane_potentials  = []
        self.firing_probabilities = []
        
        for sample_index in range(0, number_samples):
            x,m,p = self.__get_spiking_network_sample(self.dynamic_parameters["number_sources"],
                                                      self.time_meshgrid["number_time_points"])
            self.action_potentials    += [x]
            self.membrane_potentials  += [m]
            self.firing_probabilities += [p]
        
    def __get_spiking_network_sample(self, number_sources, number_time_points):
        
        epsp    = self.neuron_functions["epsp"]
        sigma   = self.neuron_functions["sigma"]
        weights = self.dynamic_parameters["connectivity_weights"]
        padding = self.neuron_functions["padding"]
        
        # set initial membrane potentials
        m = np.zeros((self.dynamic_parameters["number_sources"], number_time_points))
        
        # set initial firing probabilities
        p = np.zeros((self.dynamic_parameters["number_sources"], number_time_points))

        # set initial activities
        x = np.zeros((self.dynamic_parameters["number_sources"], number_time_points))

        # simulate neuronal dynamics at t=1,...,T 
        for tdx in range(1, number_time_points): 
            
            for ndx in range(0, number_sources):
            
                # calculate incoming currents
                preAP = np.dot(weights[:,ndx], x[:,0:tdx])
                postC = np.convolve(preAP, epsp, 'same')[tdx-1]
                
                # update membrane potentials
                m[ndx,tdx] = self.__simulate_SDE(m[ndx,tdx-1], postC)
                
                # update spiking probability
                p[ndx,tdx] = sigma(m[ndx,tdx])
            
            for ndx in range(0, number_sources):
                
                # update spike train
                zeroPadd   = (tdx < padding) or (tdx > (number_time_points-padding))
                spikeProb  = np.random.binomial(1, p[ndx,tdx])
                x[ndx,tdx] = (1-np.int(zeroPadd)) * spikeProb
            
        return x, m, p
    
    def __simulate_SDE(self, m_, inDrive):
        
        # numerically solve stochastic differential equation via Euler-Maruyama
        
        tau  = 0.015                       # membrane time constant,
        beta = 100.0*3.5                        # (input) current-to-voltage conversion,
        driving_noise  = 1.5			    # signal-to-noise relationship

        dt   = self.time_parameters["time_step"]
        
        dW = np.random.normal(loc = 0.0, scale = np.sqrt(dt))
        m  = m_ + (1/tau) * ((-m_ + beta*inDrive)*dt + driving_noise*dW)
        
        return m
    
    def ground_truth_conn(self,t,s):
        return 0.5*(np.sign(t) + 1)*t*np.exp(-t*s)

    def conn_function(self,t, connectivity_relaxation_mat, adj_mat, source, target):
        conn_dynamics = self.ground_truth_conn(t, connectivity_relaxation_mat[source, target])
        return adj_mat[source, target] * conn_dynamics / np.max(conn_dynamics)
        
    def simulate_network_dynamics(self, 
                                  ntrials_train, 
                                  ntrials_test, 
                                  params, 
                                  connectivity_relaxation = 1/0.15,
                                  AR_coefficient          = 0.6):
        adj_mat = params['network']
        time_step = params['time_step']
        time_period = params['time_period']
        connectivity_strength = params['connection_strength']
        p = adj_mat.shape[0]
        relaxation_coef         = -(AR_coefficient - 1)/time_step 
        adj_mat *= connectivity_strength
        connectivity_relaxation_mat = connectivity_relaxation * np.ones((p, p))
        connectivity_relaxation_mat[np.where(np.identity(p)==1)] = 0
        
        source_relaxation = relaxation_coef * np.ones((p,))
        MA_constants = 10**15 * np.ones((p,))
        
        self.dynamic_parameters = {"number_sources"                     : p,
                                   "connectivity_weights"               : adj_mat, # connectivity matrix of n x n
                                   "connectivity_relaxations_constants" : connectivity_relaxation_mat,
                                   "moving_average_time_constants"      : MA_constants,
                                   "relaxation_constants"               : source_relaxation}

        self.run_sampler(number_samples = ntrials_train+ntrials_test)
        t = self.time_meshgrid["time_range"]
        trials_train = self.samples[0:ntrials_train]
        trials_test = self.samples[ntrials_train:]
        ground_truth = np.zeros((int(time_period / time_step), p, p))

        for i in range(0,p):            
            for j in range(0,p):
                if i != j:
                    ground_truth[:,i,j] = self.conn_function(t, connectivity_relaxation_mat, adj_mat, i,j)
        
        return (trials_train, trials_test, ground_truth)

    def simulate_spiking_network_dynamics(self, 
                                          ntrials_train, 
                                          ntrials_test, 
                                          params):
        
        adj_mat                     = params['network']
        time_step                   = params['time_step']
        time_period                 = params['time_period']
        padding_window              = params['padding']
        
        self.time_parameters        = {"time_period": time_period,
                                       "time_step"  : time_step}
        
        connectivity_strength       = params['connection_strength']
        p                           = adj_mat.shape[0]
        adj_mat                    *= connectivity_strength
        
        self.dynamic_parameters     = {"number_sources"       : p,
                                       "connectivity_weights" : adj_mat}
        
        # get the excitatory post-synaptic potential shape
        epsp                            = self.get_epsp_kernel(time_step)
                        
        # get sigmoidal activation function                    
        sigma                           = self.get_activation_function(time_step)
        
        self.neuron_functions           = {"epsp"   : epsp,
                                           "sigma"  : sigma,
                                           "padding": padding_window}
        
        self.run_spiking_network_sampler(number_samples = ntrials_train+ntrials_test)
        
        trials_train = {"action_potentials"     : self.action_potentials[0:ntrials_train],
                        "membrane_potentials"   : self.membrane_potentials[0:ntrials_train],
                        "firing_probabilities"  : self.firing_probabilities[0:ntrials_train]}
        trials_test  = {"action_potentials"     : self.action_potentials[ntrials_train:],
                        "membrane_potentials"   : self.membrane_potentials[ntrials_train:],
                        "firing_probabilities"  : self.firing_probabilities[ntrials_train:]}
        
        return (trials_train, trials_test)
    
    def get_epsp_kernel(self,dt): 
        
        # define EPSP kernel, see e.g. Koch (1999)
        
        tau   = 0.6                                 # time constant for excitatory units
        delay = 5                                   # onset shift of the kernel
        step  = lambda arg: 1.0 * (arg > 0)         # heaviside step function

        epsp  = lambda arg: (arg**5/tau**6) * np.exp(-arg/tau)*step(arg)
        lag   = lambda arg: arg-delay
    
        length= 20                                  # EPSP duration is about 15-20 ms
        res   = dt*1000                             # temporal resolution of EPSP curve
        kernel= epsp(lag(np.arange(0,length,1.0*res)))  
        kernel= np.append(np.zeros(len(kernel)),kernel)
        kernel= 1/np.max(kernel) * kernel           # normalize kernel amplitude
        
        return kernel
    
    def get_activation_function(self,dt):
        
        fMax      = 200                          # Hz maximum firing rate
        gain      = 0.5                           # (mV)^-1 steepness of activation function
        thresh    = 7.0                         # mV constant membrane-potential shift
                        
        # declare sigmoidal activation function                    
        sigma  = lambda m: fMax * 1/(1 + np.exp(-gain*(m - thresh)))*dt
        return sigma
