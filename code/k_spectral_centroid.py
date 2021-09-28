
def main(): 
    # get time series of 200 words from lexical_change.py
    # load up the .npy that you saved
    
    k = 3
    # N = number of time series
    # ignore q!
    # randomly choose 3 time series to be initial centroids: curr_centroids C
    # curr_centroids = [None, None, None]
    # assign every time series to a centroid based on d^ (as defined in the paper), this could be a list of sets where each set contains multiple time series
    # old_centroids C^ == None
    
    # while old_centroids != curr_centroids: 
        # old_centroids = curr_centroids
        # mu = [None, None, None]
        # for j in range(k), or for every cluster
            # M = sum (over each time series in cluster j) (identity matrix - the fraction in the paper)
            # mu[j] = scipy.sparse.linalg.eigs for getting eigenvector
            # curr_centroids[j] = None
        # for i in range(N), or for every time series
            # get j*, which is the new cluster number assignment, based on min distance between time series x and each mu
            # add this time series to set C_j*, aka update the list of sets
            # update curr_centroids
            
    # to debug, you can print out cluster assignments in every iteration
    # if things still seem broken, can make fake data and cluster that
    

if __name__ == '__main__':
    main()

