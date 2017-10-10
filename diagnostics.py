import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp

def estimation_error(ground_truth, prediction):
    mse = lambda x, y: np.mean(np.power(x-y, 2))    
    ntrials,p,_,n = prediction.shape    
    mse_scores = []    
    for trial in range(0, ntrials):
        for i in range(0, p):
            for j in range(0, p):
                if i != j:
                    x = ground_truth[:,i,j]
                    y = prediction[trial,i,j,:]
                    mse_scores += [mse(x, y)]
    return mse_scores
#
def plot_distances(edges, centroids):
    plt.figure(figsize=(4,4))
    edges = edges.T
    distance_to_centroid = sp.spatial.distance.cdist( edges, centroids )
    closest_centroid = np.argmin( distance_to_centroid, axis=1 )        
    closest_centroid2color = lambda c: 'b' if c>0 else 'r'     
    plt.scatter( distance_to_centroid[:,0], 
                distance_to_centroid[:,1], 
                c=map( closest_centroid2color, closest_centroid ) )
    plt.xlabel('Distance to first centroid')
    plt.ylabel('Distance to second centroid')
    plt.title('Distances of edges to edge centroids.')
    plt.savefig('edge2centroid_distances.pdf')
#    
def plot_scale_fit(second_moment_matrices, scale_matrix, freq_range, freq_bound):
    
    frequency_filter = lambda freq, freq_bound: ((freq > -freq_bound) & (freq < freq_bound))
    normalize = lambda x: x/np.sum(x)
    smoothing_function = lambda x,l: np.exp(-np.power(x,2)/(2*l**2))/(np.sqrt(np.pi*2*l**2))
    
    p = len(second_moment_matrices)
    plt.figure(figsize=(10,7))
    
    for i in range(0,p):
        for j in range(0,p):
            if i != j:
                plt.subplot(p, p, i * p + j + 1)
                frequencies, kernel = zip(*[(freq, val) 
                                            for (freq,val) in zip(freq_range, second_moment_matrices[i][j]) 
                                            if frequency_filter(freq, freq_bound)])
                ax = plt.gca()
                ax.plot(frequencies, normalize( kernel ), label='Empirical kernel')
                ax.plot(frequencies, smoothing_function( frequencies, scale_matrix[i][j] ), label='Gaussian fit')
                #ax.set_xlabel('Frequencies')
                #ax.set_ylabel('Magnitude')
                #ax.set_title('1 / scale = {:.3f}'.format(1 / scale_matrix[i][j]))
    plt.legend(bbox_to_anchor=(1.5, 0.5), loc='upper center', borderaxespad=0.)
    plt.suptitle('Gaussian distribution fitted to spectrum to determine scale (i.e. temporal smoothing).')
    plt.savefig('scale_fit.pdf')
#
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
#
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
                #plt.plot(time_range[plotrange], ground_truth[plotrange, i, j], label='Ground truth', color='r')
                ax = plt.gca()
                mean = np.mean(connectivity[:, i, j, plotrange], axis=0)
                std = np.std(connectivity[:, i, j, plotrange], axis=0)
                intv = 1.96 * std / np.sqrt(ntrials)
                plt.plot(time_range[plotrange], mean, color='green', label='GP-CaKe')
                ax.fill_between(time_range[plotrange], mean - intv, mean + intv, facecolor='green', alpha=0.2)
                ax.axis('tight')
                ax.axvline(x=0.0, linestyle='--', color='black', label='Zero lag')
                ax.set_xlim([t0, 2.0])
                #ax.set_ylim([ylim_min, ylim_max])
                ax.set_xlabel('Time lag')
                ax.set_ylabel('Connectivity amplitude')
    plt.legend(bbox_to_anchor=(1.05, 0), loc='upper center', borderaxespad=0.)
    plt.suptitle('Mean connectivity')
    plt.draw()    
    
