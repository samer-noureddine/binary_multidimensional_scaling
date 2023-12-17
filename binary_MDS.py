# implementing the greedy max cut algorithm from Rohde, 2002
import numpy as np
from scipy import stats, spatial
import os
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(1)

with open('./word2vec_semfeatmatrix.npy','rb') as f:
    w2vmatrix = np.load(f)
with open(r'./1579words_words.txt') as f:
    w = f.read()
    wordlist = w.split('\n')

def upper_triangular_matrix(N):
    upper_triangular = np.zeros((N, N), dtype=int)
    upper_triangular[np.triu_indices(N, k=1)] = np.arange(np.triu_indices(N, k=1)[0].shape[0])
    return upper_triangular

def all_but_index_i(np_idx_array,i):
    ind = np.ones(np_idx_array.shape[0], bool)
    ind[i] = False
    return np_idx_array[ind]

def hamming_distance(bmatrix):
    # return spatial.distance.pdist(bmatrix.T, metric = 'hamming')*bmatrix.shape[0] #d_ij
    # return spatial.distance.pdist(bmatrix, metric = 'hamming')*bmatrix.shape[1]
    return spatial.distance.pdist(bmatrix, metric = 'cityblock')

def greedy_max_cut(target_matrix,N,D):
    binary_matrix = np.zeros((N,D))
    d = hamming_distance(binary_matrix)
    t = np.corrcoef(target_matrix.T)[np.triu_indices(target_matrix.shape[1],1)]
    cost = []
    cost.append(np.sum((d-t)**2))
    t = (0.5 - 0.5*t)*binary_matrix.shape[1]
    SECONDARY_ADJUSTMENTS = 4
    PRIMARY_ADJUSTMENTS = 1
    index_matrix = upper_triangular_matrix(N)
    index_matrix = np.triu(index_matrix)+np.tril(index_matrix.T) # make it a symmetric matrix

    for sec_adjust in range(SECONDARY_ADJUSTMENTS):
        primary_adjustment = True
        if sec_adjust != 0:
            primary_adjustment = False

        for col_k in range(binary_matrix.shape[1]):
            for iteration in range(PRIMARY_ADJUSTMENTS):
                for i in range(binary_matrix.shape[0]):
                    if iteration == 0 and i == 0 and primary_adjustment:
                        binary_matrix[i,col_k] = np.random.randint(0,2)
                    else:
                        # if first primary adjustment, focus only on past items
                        if iteration == 0 and primary_adjustment:
                            distance_indices_that_matter = all_but_index_i(index_matrix[:,i],i)[:i]
                            a_matrix = binary_matrix[:i,col_k]
                        else:
                            distance_indices_that_matter = all_but_index_i(index_matrix[:,i],i)
                            a_matrix = all_but_index_i(binary_matrix[:,col_k],i)
                        a = 2*a_matrix - 1 
                        b = 2*(d[distance_indices_that_matter] - t[distance_indices_that_matter])+1
                        
                        c0_minus_c1_efficient = a @ b
                        if c0_minus_c1_efficient < 0:
                            binary_matrix[i,col_k] = 0
                        else:
                            binary_matrix[i,col_k] = 1
                    d = hamming_distance(binary_matrix)
                    cost.append(np.sum((d-t)**2))

    return binary_matrix,cost

N = 16
D = 512

semantic_target_matrix = w2vmatrix[:,:N]
new_binary_matrix,cost = greedy_max_cut(semantic_target_matrix,N,D)
t = np.corrcoef(semantic_target_matrix.T)[np.triu_indices(semantic_target_matrix.shape[1],1)]
t = 0.5 - 0.5*t
t = t*new_binary_matrix.shape[1]
final_d = hamming_distance(new_binary_matrix)

folder_path = f'./plots/N_{N}_D_{D}/'
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

index_matrix = upper_triangular_matrix(N)
img_matrix_d = np.zeros(index_matrix.shape)
img_matrix_d[np.triu_indices(N, k=1)] = final_d
img_matrix_d = np.triu(img_matrix_d)+np.tril(img_matrix_d.T)
plt.imshow(img_matrix_d,cmap='viridis')
plt.colorbar()
plt.savefig(f'./{folder_path}/binary_matrix_distances_sparsity.png')
plt.close()

img_matrix_t = np.zeros(index_matrix.shape)
img_matrix_t[np.triu_indices(N, k=1)] = t
img_matrix_t = np.triu(img_matrix_t)+np.tril(img_matrix_t.T) # make it a symmetric matrix
plt.imshow(img_matrix_t,cmap='viridis')
plt.colorbar()
plt.savefig(f'./{folder_path}/target_matrix_distances.png')
plt.close()


'''
Some issues.

The semantic matrix is not sparse. It is half 1, half 0.

Also, the algorithm is slow. It will take a very long time to run 1000 words.

'''
