import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import random as rnd
from chainer.training import extensions


# In[3]:
class convolutional_layer(ChainList):
            def __init__(self, number_features, dilation_factor, nobias):
                self.dilation_factor = dilation_factor
                super(convolutional_layer, self).__init__(
                # the size of the inputs to each layer will be inferred
                    L.DilatedConvolution2D(in_channels=number_features, 
                                           out_channels=number_features, 
                                           ksize=(3,1), 
                                           dilate = dilation_factor, 
                                           pad = (dilation_factor,0), 
                                           initialW=chainer.initializers.HeNormal(),
                                           nobias=nobias)
                    )

            def __call__(self, x):
                h = F.relu(self[0](x))
                return h 

class convolutional_MLP(ChainList):
    def __init__(self, 
                 number_features, 
                 number_center_points, 
                 receptive_fields_size, 
                 number_hidden_units, 
                 number_output_kernels, 
                 number_target_time_points, 
                 residual = 1,
                 nobias = "False"):
        ## private classes ##
        
        ## private functions ##
        def get_dilation_factors(receptive_fields_size):
            dilation_factors = [1]
            index = 0
            while 3 + 2*sum(dilation_factors) < receptive_fields_size:
                index += 1
                dilation_factors += [2**index]
            return dilation_factors[:-1], 3 + 2*sum(dilation_factors[:-1])
        ## main code ##
        self.number_output_kernels = number_output_kernels
        self.number_target_time_points = number_target_time_points
        self.residual = residual
        dilation_factors = get_dilation_factors(receptive_fields_size)[0]
        links = [L.Convolution2D(in_channels=1, 
                                 out_channels=number_features, 
                                 ksize=(3,1), 
                                 pad=(1,0), 
                                 initialW=chainer.initializers.HeNormal(scale=15.0),
                                 nobias=nobias)]
        for dilation in dilation_factors:
            links += [convolutional_layer(number_features, 
                                          dilation,
                                          nobias = nobias)]
        num_conv = len(links)
        print "Number of convolutional layers: " + str(num_conv)
        links += [L.Linear(in_size=None, 
                           out_size=number_hidden_units, 
                           nobias=nobias)]
        num_lin = 1
        print "Number of linear layers: " + str(num_lin)
        for output_point_index in range(0,number_target_time_points):
            for kernel_out_index in range(0, self.number_output_kernels):
                links += [L.Linear(in_size=None, 
                                   out_size=number_center_points, 
                                   nobias=nobias)]
        num_out = len(links) - num_lin - num_conv
        print "Number of output links: " + str(num_out)
        print "total number of links: " + str(len(links))
        super(convolutional_MLP, self).__init__(*links)

    def __call__(self, x):
        h = F.relu(self[0](x))
        for layer_index in range(1,len(self)-self.number_output_kernels*self.number_target_time_points - 1):
            h = self[layer_index](h) + self.residual*h
        h = F.relu(self[layer_index + 1](h))
        #h = F.relu(self[layer_index + 2](h))
        out_list = []
        for output_point_index in range(0,self.number_target_time_points):
            point_out_list = []
            for kernel_out_index in range(0, self.number_output_kernels):
                out_index = output_point_index*self.number_output_kernels + kernel_out_index + 1
                point_out_list += [F.softplus(self[-out_index](h))]
                #point_out_list += [F.math.basic_math.pow(F.relu(self[-out_index](h)),2)]
                #point_out_list += [F.elu(self[-out_index](h), alpha) + alpha]
            out_list += [point_out_list]
        return out_list


# In[111]:

class neural_ensemble_forecaster(object):
    #
    def __init__(self, 
                 network_parameters,
                 kernels_parameters,
                 center_points_parameters,
                 sampler):
        ## private functions ##
        def get_center_points(train_target, delta):
            ## private functions ##
            def trim_center_points(sorted_center_points, delta):
                trimmed_center_points = [sorted_center_points[0]]
                for index in range(0,len(sorted_center_points) - 1):
                    if np.abs(sorted_center_points[index + 1] - trimmed_center_points[-1]) >= delta:
                        trimmed_center_points += [sorted_center_points[index + 1]]
                return np.array(trimmed_center_points)
            ## main code ##
            flattened_target = train_target.flatten()
            sorted_target = np.sort(flattened_target)
            center_points = trim_center_points(sorted_target, delta)
            return center_points
        ## main code ##
        self.description = "This class contains the tools for using the neural ensemble forecaster"
        initial_data, initial_target = sampler(center_points_parameters["initial_data_size"])
        self.target_scale = np.sqrt(np.var(initial_target))
        self.number_target_time_points = network_parameters["number_target_time_points"]
        self.center_points = get_center_points(initial_target, 
                                               self.target_scale*kernels_parameters["minimal_scale"])
        self.basic_kernels = self.__get_basic_kernels_list(kernels_parameters["number_output_kernels"], 
                                                           self.target_scale*kernels_parameters["minimal_scale"], 
                                                           self.target_scale*kernels_parameters["scale_step"])
        self.initialize_network(network_parameters)
        self.sampler = sampler
    #
    def initialize_network(self, 
                           network_parameters):
        self.network = convolutional_MLP(network_parameters["number_features"], 
                                         len(self.center_points), 
                                         network_parameters["receptive_fields_size"], 
                                         network_parameters["number_hidden_units"], 
                                         len(self.basic_kernels), 
                                         network_parameters["number_target_time_points"])
        self.optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
        self.optimizer.setup(self.network) 
    #
    def __get_output_matrices(self,
                              target_points):
        try:
            target_mesh_1, target_mesh_2 = np.meshgrid(self.center_points.flatten(), target_points.data.flatten())
        except AttributeError:
            target_mesh_1, target_mesh_2 = np.meshgrid(self.center_points.flatten(), target_points.flatten())
        output_matrices = []
        for basic_kernel in self.basic_kernels:
            output_matrices += [chainer.Variable(basic_kernel(target_mesh_1, target_mesh_2).astype("float32"))]
        return output_matrices
    #
    def eveluate_output_pdf(self,
                            data,
                            output_range,
                            output_index):
        NN_output = self.network(chainer.Variable(data[:,None,:,None]))
        output_matrices = self.__get_output_matrices(output_range)
        output_weights_list = []
        for kernel_index in range(0,len(output_matrices)):
            output_weights_list += [np.matrix(NN_output[output_index][kernel_index].data[:,None]).T]
        nn_output_pdf = 0
        normalization = 0
        matrix_index = 0
        active_weights_count = 0
        weights_count = 0
        for output_matrix in output_matrices:
            nn_output_pdf += np.matrix(output_matrix.data)*output_weights_list[matrix_index]
            normalization += float(np.sum(output_weights_list[matrix_index]))
            active_weights_count += int(np.sum(output_weights_list[matrix_index] > 0))
            weights_count += int(np.sum(output_weights_list[matrix_index] > -1))
            matrix_index += 1
        return np.array(nn_output_pdf).flatten()/normalization
    #
    def evaluate_average_likelihood(self,
                                    data, 
                                    target,
                                    target_time):
        target_length = len(target_time)
        likelihood_list = [self.eveluate_output_pdf(data, target.flatten()[index], index) 
                           for index in range(0, target_length)]
        return np.mean(likelihood_list)
    #
    def __get_basic_kernels_list(self,
                                 number_kernels, 
                                 minimal_scale, 
                                 scale_step):
        basic_kernels_list = []
        scale = minimal_scale
        kernel_function = lambda x,y,scale: (1/(np.sqrt(2*np.pi)*scale))*np.exp(-0.5*(x - y)**2/scale**2)
        for kernel_index in range(0, number_kernels):
            scale += kernel_index*scale_step
            basic_kernels_list += [lambda x,y,scale = scale: kernel_function(x,y,scale)]
        return basic_kernels_list 
    #
    def get_forecast(self,
                     data, 
                     target_time, 
                     output_range):
            target_length = len(target_time)
            pdf_array = np.zeros(shape = (target_length,len(output_range)))
            for index in range(0,target_length):
                pdf_array[index,:] = self.eveluate_output_pdf(data,
                                                              output_range,  
                                                              index)
            self.forecast = pdf_array
            self.output_range = output_range
            self.target_time = target_time
            return pdf_array 
    #
    def compute_performance(self, target, radius, time_index):
        normalized_probability = self.forecast[time_index,:]/np.sum(self.forecast[time_index,:])
        probability_outcome_pairs = zip(normalized_probability, 
                                        self.output_range)
        probability = np.sum([pair[0]
                              for pair in probability_outcome_pairs
                              if np.abs(pair[1] - target[0][time_index]) < radius])
        return probability
    #
    def visualize_forecast(self, 
                           data_time, 
                           data, 
                           target_time, 
                           output_range, 
                           **kwargs):
        forecast = self.get_forecast(data, target_time, output_range)
        import matplotlib.gridspec as gridspec
        fig, axs = plt.subplots(1,2,figsize=(6*2,6*1))
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[(max(data_time) - min(data_time)),(max(target_time) - min(target_time))],
                               height_ratios=[4,1])
        gs.update(wspace=0.00001)

        ax1 = plt.subplot(gs[0])
        plt.scatter(data_time, data.flatten())
        plt.xlim(min(data_time),max(data_time))
        plt.ylim(min(output_range), max(output_range))

        ax2 = plt.subplot(gs[1])
        from matplotlib.colors import LogNorm
        plt.imshow((np.transpose(forecast)), 
                   aspect='auto',
                   origin='lower',
                   extent=(min(target_time),max(target_time), min(output_range),max(output_range)),
                   norm=LogNorm(vmin=0.01, vmax=2), 
                   cmap=plt.get_cmap("Reds")) 
        plt.ylim(min(output_range), max(output_range))
        if "target" in kwargs.keys():
            plt.plot(target_time, kwargs["target"].flatten(), color = "k", lw = 2, ls = "--")
        # get all the labels of this axis
        from matplotlib.ticker import MaxNLocator
        ax2.xaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax2.set_yticklabels([])
    #
    def train_network(self,
                      n_epochs, 
                      batch_size):
        ## private functions ##
        def nonparametric_cost(weights_list, output_matrices):
            ## private functions
            def nonparametric_log_likelihood(weights_list, output_matrices, jitter):
                normalization = 0.
                nn_likelihood = 0.
                number_datapoints = output_matrices[0].shape[0]
                for kernel_index in range(0, len(output_matrices)):
                    nn_likelihood += F.math.sum.sum(jitter + F.math.basic_math.mul(output_matrices[kernel_index],weights_list[kernel_index]), axis = 1)
                    normalization += F.math.sum.sum(jitter + weights_list[kernel_index], axis = 1)
                log_likelihood = F.math.sum.sum(F.log(nn_likelihood) - F.log(normalization))
                return log_likelihood
            ## main code ##
            log_likelihood = nonparametric_log_likelihood(weights_list, output_matrices, jitter = 10**-30)
            return -log_likelihood
        def get_train_target(targets_array, target_point):
            train_target = []
            batch_size = np.shape(targets_array)[0]
            for index in range(0, batch_size):
                train_target += [float(targets_array[index, target_point])]
            return chainer.Variable(np.array(train_target))  
        ## main code ##
        test_loss_list = []
        filter_parameter = 0.1
        for iteration in range(n_epochs):
            # Sample batch
            loss = 0
            data, targets = self.sampler(batch_size)
            batch_train_data = chainer.Variable(data[:,None,:,None])
            # Get the result of the forward pass. 
            output = self.network(batch_train_data)
            for target_point in range(0, self.number_target_time_points):
                train_target = get_train_target(targets,
                                                target_point)
                # Get the result of the forward pass. 
                weights_list = output[target_point]
                # Get the output matrices
                output_matrices = self.__get_output_matrices(train_target)
                # Calculate the loss between the training data and target data.
                loss += nonparametric_cost(weights_list, output_matrices)
            # Zero all gradients before updating them.
            self.network.zerograds()

            # Calculate and update all gradients.
            loss.backward()

            # Use the optmizer to move all parameters of the network
            # to values which will reduce the loss.
            self.optimizer.update()

            if iteration == 0:
                filtered_loss = loss.data/float(batch_size)
            else:
                filtered_loss = (1 - filter_parameter)*filtered_loss + filter_parameter*loss.data/float(batch_size)
            if iteration%2 == 0:
                #print iteration
                print "Training set loss: " + str(filtered_loss)
                test_loss_list += [filtered_loss]
        return test_loss_list
