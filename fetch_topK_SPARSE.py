'''
Created on Jul 7, 2022
@note Fetches topK similar nodes to each node from a similarity matrix 
@author: masoud
'''
import numpy as np
import math
import tensorflow as tf

def get_listMLE_topK(result_matrix, topK):    
    '''
        @return: top_indices: |V|*topK matrix contains indices of topK nodes to each node
    '''      
    top_indices = np.zeros((result_matrix.shape[0],topK),dtype='int32')
    for target_node in range (0,result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node,:].toarray()[0], axis=0)[::-1][:topK]  ## sorting the indices on descending order of similarity values and return the topk
        top_indices[target_node] = target_node_res_sorted.copy()
    print ("Top {} similar nodes are fetched ... ".format(topK-1))
    return top_indices

