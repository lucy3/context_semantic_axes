import random
import numpy as np 
from collections import defaultdict
from numpy import linalg as LA
from copy import deepcopy

# ignore q!

def main(): 
    
    def assign_clusters(centroids, cluster_array):
        '''
        @input
        - centroids: a list of arrays of numbers
        - cluster_array: a list of arrays of numbers (all data points)

        @output
        - a list of indices into centroids list
        '''
        clusters = []
        for i in range(len(cluster_array)):
            distances = []
            for centroid in centroids:
                distances.append(d_hat(centroid, cluster_array[i]))
            cluster = [z for z, val in enumerate(distances) if val==min(distances)]
            clusters.append(cluster[0])
        return clusters
    
    # get time series of 200 words from lexical_change.py
    # load up the .npy that you saved
    
    # some fake data
    fake1 = [2, 4, 16, 256, 65536]
    fake2 = [65546, 256, 16, 4, 2]
    fake3 = [0, 5, 10, 15, 20]
    fake4 = [25, 20, 15, 10, 5]
    fake5 = [3, 9, 81, 6561, 81]
    fake6 = [9, 81, 6561, 81, 9]
    fake7 = [20, 40, 160, 2560, 655360]
    fake8 = [655360, 2560, 160, 40, 20]
    fake9 = [0, 50, 100, 150, 200]
    fake10 = [250, 200, 150, 100, 50]
    fake11 = [30, 90, 810, 65610, 810]
    fake12 = [90, 810, 65610, 810, 90]
    
    # list of lists
    fake_list = [fake1, fake2, fake3, fake4, fake5, fake6, fake7, fake8, fake9, fake10, fake11, fake12]
    # N = number of time series
    N = len(fake_list)
    for i in range(N): 
        fake_list[i] = np.array(fake_list[i])
    
    k = 3
    
    curr_centroids = [None] * k
    
    # randomly choose 3 time series to be initial centroids: curr_centroids C
    # list of 3 time series lists
    curr_centroids = random.sample(fake_list, 3)
    
    # optimal alpha, as defined in the paper, but unsure about this
    alpha = lambda x, y: (np.dot(x, y)) / (np.linalg.norm(y))**2
    
    # d^ as defined in the paper
    d_hat = lambda x, y: (np.linalg.norm(x - alpha(x,y)*y)) / (np.linalg.norm(x))
   

    # assign every time series to a centroid based on d^ 
    # each value in this list is the cluster number of a vector
    clusters = assign_clusters(curr_centroids, fake_list)
    
    print(clusters)
    
    # a mapping from cluster number to vector indices
    # C^
    clust_vec_idx = defaultdict(set)
    for i, j in enumerate(clusters): 
        clust_vec_idx[j].add(i)

    it = 0
    while True:
        print("Iteration", it)
        print(clust_vec_idx)
        it += 1
        old_clust_vec_idx = deepcopy(clust_vec_idx)
        mu = [None] * k
        for j in clust_vec_idx:
            fraction = lambda x: np.dot(x, np.transpose(x)) / np.linalg.norm(x)**2
            M = np.zeros((fake_list[0].shape[0], fake_list[0].shape[0]))
            for vec_idx in clust_vec_idx[j]: 
                vec = fake_list[vec_idx]
                M += np.identity(vec.shape[0]) - fraction(vec)
            w, v = LA.eig(M)
            idx = np.argmin(w)
            mu[j] = v[idx]
            clust_vec_idx[j] = set()
            
        for vec_idx in range(N): 
            vec = fake_list[vec_idx]
            distances = []
            for j in range(len(mu)): 
                distances.append(d_hat(vec, mu[j]))
            j_star = np.argmin(distances)
            clust_vec_idx[j_star].add(vec_idx)
            
        same = True
        for j in old_clust_vec_idx:
            if old_clust_vec_idx[j] != clust_vec_idx[j]:
                same = False
        if same: 
            break
            
            
            


            
    # to debug, you can print out cluster assignments in every iteration
    # if things still seem broken, can make fake data and cluster that
    

if __name__ == '__main__':
    main()

