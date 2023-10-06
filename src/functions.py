import numpy as np
from fastdtw import fastdtw
import math
from scipy.spatial.distance import euclidean, sqeuclidean, cosine, correlation, chebyshev, cityblock, minkowski

template = np.array([[1,1,3,6,12], [1,1,2,5,11], [3,3,3,3,3]])
input_arr = np.array([[1,2,11], [1,3,12], [3,3,3]])


def compute_cost_matrix(input_array, template):    
    distance_matrix = np.zeros((len(template), len(input_array)))
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[0])):
            distance_matrix[i][j] = eucledian(input_array[j], template[i]) 
   
    return distance_matrix

def eucledian(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i]-b[i])**2
    return math.sqrt(total)

def compute_accumulated_cost_matrix(C):
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

C = compute_cost_matrix(input_array=input_arr.T, template=template.T)
D =  compute_accumulated_cost_matrix(C)
print('Accumulated cost matrix D =', D, sep='\n')
print('DTW distance DTW(X, Y) =', D[-1, -1])

print("DTW calculation using library:", fastdtw(input_arr.T, template.T, dist=euclidean))
