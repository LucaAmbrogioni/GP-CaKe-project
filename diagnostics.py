import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


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
                ax.set_xlabel('Frequencies')
                ax.set_ylabel('Magnitude')
                ax.set_title('1 / scale = {:.3f}'.format(1 / scale_matrix[i][j]))
    plt.legend(bbox_to_anchor=(1.5, 0.5), loc='upper center', borderaxespad=0.)
    plt.suptitle('Gaussian distribution fitted to spectrum to determine scale (i.e. temporal smoothing).')
    
    
    