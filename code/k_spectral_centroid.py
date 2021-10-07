import random
import numpy as np 
from collections import defaultdict
from numpy import linalg as LA
from copy import deepcopy

# ignore q!

def main(): 
    
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
    k = 3
    fake_list = np.array(fake_list)
    
    # cluster membership for each time series
    mem = np.array([random.randint(0, k-1) for idx in range(N)])
    
    # optimal alpha, as defined in the paper
    alpha = lambda x, y: np.dot(x, y) / np.dot(y, y)
    
    # d^ as defined in the paper
    d_hat = lambda x, y: (np.linalg.norm(x - alpha(x,y)*y)) / (np.linalg.norm(x))
    
    fraction = lambda x: np.dot(x, np.transpose(x)) / np.linalg.norm(x)**2
    

    for it in range(100):
        print("Iteration", it)
        prev_mem = deepcopy(mem)
        mu = np.zeros((k, fake_list[0].shape[0]))
        for j in range(k):
            M = np.zeros((fake_list[0].shape[0], fake_list[0].shape[0]))
            for vec_idx in range(N): 
                if mem[vec_idx] == j: 
                    vec = fake_list[vec_idx]
                    M += np.identity(vec.shape[0]) - fraction(vec)
            w, v = LA.eig(M)
            idx = np.argmin(w)
            if np.sum(v[idx]) < 0: 
                mu[j] = -v[idx]
            else: 
                mu[j] = v[idx]

        for vec_idx in range(N): 
            vec = fake_list[vec_idx]
            distances = []
            for j in range(k): 
                distances.append(d_hat(vec, mu[j]))
            j_star = np.argmin(distances)
            mem[vec_idx] = j_star
            
        print(mem)
        if np.linalg.norm(prev_mem - mem) == 0: 
            break

    

if __name__ == '__main__':
    main()