#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

#STREAM_SIZE = 100		#size of each piece of data 
DIMENSION = 500			#dimension of input data point
K_CLUSTER = 100			#number of clusters
# ALPHA = 0.5			#learning rate

def emit(means):
    """Emit the means vector from one stream"""
    for ii in range(means.shape[0]):
        ls = means[ii, :]
        ls = map(lambda l: str(l), ls)
        print(' '.join(ls))


def parse(line):
    """Parse line into weight vector"""
    line = line.strip().split("\t")
    arr = []
    for w in line:
        arr.append(float(w))

    return np.asarray(arr)


def sqr_distance(x, y):
    sqr_dist = np.zeros(shape=(y.shape[0]))

    for ii in range(y.shape[0]):
        sqr_dist[ii] = np.power(np.linalg.norm(np.subtract(x, y[ii,:])),2)
    
    return sqr_dist


def sqr_distance_pro(x, y, sqr_pro):
    sqr_dist = sqr_distance(x, y)
    
    sqr_pro = np.add(sqr_pro, sqr_dist / (np.sum(sqr_dist)))	
    sqr_pro = sqr_pro / (np.sum(sqr_pro))
    
    return sqr_pro


def init_cluster_center(x, K_CLUSTER):
    """To compute the initial clustering center for the center"""
    #N = x.shape[0]
    clu_center = np.zeros(shape=(K_CLUSTER, DIMENSION))
    #generate the first clustering center randomly
    
    first_clu_index = np.random.randint(x.shape[0])
    clu_center[0, :] = x[first_clu_index, :]

    x = np.delete(x,(first_clu_index),axis = 0)
    #select other clustering center
    num_selcted_clu = 1
    
    sqr_pro = np.zeros(shape=(x.shape[0]))
    
    while True:
        sqr_pro = sqr_distance_pro(clu_center[num_selcted_clu-1,:],x, sqr_pro)	
        clu_index_temp = np.random.choice(x.shape[0], 1, p = sqr_pro)
        clu_center[num_selcted_clu,:] = x[clu_index_temp,:]
        num_selcted_clu +=1
        first_clu_index=clu_index_temp
        x = np.delete(x,(first_clu_index),axis = 0)
        sqr_pro = np.delete(sqr_pro, (first_clu_index), axis = 0)
        
        if (num_selcted_clu >= K_CLUSTER):
             break
        
    return clu_center


def seq_k_means(x, y):
    num_selected_data = np.ones(K_CLUSTER)

    for t in range(y.shape[0]):
        sqr_dist = sqr_distance(y[t,:],x)
        min_dex = np.where(sqr_dist == sqr_dist.min())
        ALPHA = 1 / (num_selected_data[int(min_dex[0])] + 1)
        x[int(min_dex[0]), :] += ALPHA*(np.subtract(y[t,:],x[int(min_dex[0]), :]))
        num_selected_data[int(min_dex[0])] += 1
    return x


if __name__ == "__main__":
    # pr.enable()
    stream = np.zeros(shape=(0, DIMENSION))      # Initialise batch matrix

    for line in sys.stdin:
        line = line.strip()            
        x_original = parse(line)        
        x_original.shape = (1, DIMENSION)
        stream = np.concatenate((stream, x_original))
        

        #find the inital clustering center by k means ++
    if stream.shape[0] <= K_CLUSTER:
        emit(stream)

        # print(stream)
    if stream.shape[0] > K_CLUSTER:                   
        initial = init_cluster_center(stream, K_CLUSTER)
        #using sequence k-means
        res_center = seq_k_means(initial, stream)
        emit(res_center)

        # print(res_center)
