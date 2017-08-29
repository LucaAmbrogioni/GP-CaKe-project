
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.setrecursionlimit(20000)

def estimation_error(ground_truth, connectivity):
    mse = lambda x, y: np.mean(np.power(x-y, 2))
    
    ntrials,p,_,n = connectivity.shape
    
    mse_scores = []
    
    for trial in range(0, ntrials):
        for i in range(0, p):
            for j in range(0, p):
                if i != j:
                    x = ground_truth[:,i,j]
                    y = connectivity[trial,i,j,:]
                    mse_scores += [mse(x, y)]
    return mse_scores
            
    

def plot_connectivity(ground_truth, connectivity, time_range, t0):
    ylim_max = 1.2 * np.max(ground_truth)
    ylim_min = -1.0 * np.max(ground_truth)
    x0 = np.where(time_range < t0)[0][-1]
    n = ground_truth.shape[0]
    plotrange = np.arange(x0, n, 1)
    (ntrials,p,_,_) = connectivity.shape

    plt.figure(figsize=(12,8))
    for i in range(0, p):
        for j in range(0, p):
            if i != j:
                plt.subplot(p, p, i * p + j + 1)
                plt.plot(time_range[plotrange], ground_truth[plotrange, i, j], label='Ground truth', color='r')
                ax = plt.gca()
                mean = np.mean(connectivity[:, i, j, plotrange], axis=0)
                std = np.std(connectivity[:, i, j, plotrange], axis=0)
                intv = 1.96 * std / np.sqrt(ntrials)
                plt.plot(time_range[plotrange], mean, color='green', label='GP-CaKe')
                ax.fill_between(time_range[plotrange], mean - intv, mean + intv, facecolor='green', alpha=0.2)
                ax.axis('tight')
                ax.axvline(x=0.0, linestyle='--', color='black', label='Zero lag')
                ax.set_xlim([t0, 2.0])
                ax.set_ylim([ylim_min, ylim_max])
                ax.set_xlabel('Time lag')
                ax.set_ylabel('Connectivity amplitude')
    plt.legend(bbox_to_anchor=(1.05, 0), loc='upper center', borderaxespad=0.)
    plt.suptitle('Mean connectivity')
    plt.draw()

def plot_samples(samples):
    nsamples = len(samples)
    (p,_) = samples[0].shape

    plt.figure(figsize=(nsamples*5, 4))
    for i in range(0, nsamples):
        plt.subplot(1,nsamples,i+1)
        for j in range(0, p):
            plt.plot(np.array(samples)[i,j,:], label='Node {:d}'.format(j+1))
        plt.xlabel('time (ms / 10)')
        plt.ylabel('signal amplitude')
        plt.title('Sample {:d}'.format(i+1))
        plt.legend()
    plt.suptitle('A few selected trials')
    plt.draw()

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
            print("The side should be either left or right")
            return
    else:
        M = np.matrix(divider)
        if side is "right":
            result = np.linalg.solve(M.T,X.T).T
        elif side is "left":
            result = np.linalg.solve(M,X)
        else:
            print("The side should be either left or right")
            return
    return result

def heaviside(x):
    return 0.5*(np.sign(x) + 1)
    
def step(x, step_point, side):
    if side is "right":
        s = +1
    elif side is "left":
        s = -1
    return heaviside(s*x - s*step_point)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x)>0)

def nested_zip(*list_matrices):
    zipped_matrix = []
    for first_index,_ in enumerate(list_matrices[0]):
        rows_list = []
        for matrix in list_matrices:
            rows_list += [matrix[first_index]]
        zipped_row = zip(*rows_list)
        zipped_matrix += [zipped_row]
    return zipped_matrix  

def fill_diagonal(matrix, entry):    
    filled_matrix = [deepcopy(row) for row in matrix]    
    for index in range(0, len(filled_matrix)):
        filled_matrix[index][index] = entry
    return filled_matrix

def nested_map(function, matrix):
    return [map(function, row) for row in matrix]

def nested_reduce(function, matrix):
    return reduce(function, [reduce(function,row) for row in matrix])

def foldRight(function, initial_value, lst):
    if len(lst) == 0:
        return initial_value
    else:
        return function(lst[0], foldRight(function, initial_value, lst[1:]))

def nested_foldRight(first_function, second_function, initial_value, matrix):
    return reduce(second_function, [foldRight(first_function, initial_value, row) for row in matrix])
    


