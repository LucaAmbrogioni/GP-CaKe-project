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

    def ground_truth_conn(self,t,s):
       return 0.5*(np.sign(t) + 1)*t*np.exp(-t*s)

    def conn_function(self,t, connectivity_relaxation_mat, adj_mat, source, target):
        conn_dynamics = self.ground_truth_conn(t, connectivity_relaxation_mat[source, target])
        return adj_mat[source, target] * conn_dynamics / np.max(conn_dynamics)
        
    def simulate_network_dynamics(self, ntrials_train, ntrials_test, params):
        adj_mat = params['network']
        time_step = params['time_step']
        time_period = params['time_period']
        connectivity_strength = params['connection_strength']
        p = adj_mat.shape[0]
        
        AR_coefficient          = 0.6
        relaxation_coef         = -(AR_coefficient - 1)/time_step 
        connectivity_relaxation = 1/0.15 
        
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
