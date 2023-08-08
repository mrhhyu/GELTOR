'''
Created on July 7, 2022

Implementation of AdaSim* for using in graph embeddings.
This Implementation can also be used to compute only AdaSim* scores (i.e., "compute_only_AdaSim_star")

@author: masoud
'''
import numpy as np
import networkx as nx
import math
from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.preprocessing import normalize
import fetch_topK_SPARSE as prData

def compute_AdaSim_star (graph='', iterations=0, damping_factor=0.8, topK=0, loss=''):
    '''
        @param loss: indicates the loss function
            1- loss='listMLE', the results are returned for ListMLE loss:
                top_indices is a |V|*|V| matrix contains the rank of each node regarding to a target node;
                top_simvals is a |V|*|V| matrix contains the similarity score of each node regarding to a target node
                
            2- loss='listMLE_topK', the results are returned for TopK_ListMLE loss: 
                top_indices is a |V|*topK matrix contains indices of topK nodes to each node
                top_simvals is a |V|*topK matrix contains similarity values of topK nodes to each node
                mask_ is a |V|*|V| matrix contains 'False' for topK similar nodes and 'True' for other nodes
                
            3- loss='at_Rank', the results are returned for Attention Rank loss: 
                top_indices is a |V|*2*topK matrix contains [indices of topK nodes to each node] + [labels values 0, ..., Topk-1 normalized in range [0,1]]
                top_simvals is a |V|*topK matrix contains similarity values of topK nodes to each node
            
            4- loss='listNET', the result is returned for listNET loss:
                result_matrix is a |V|*|V| matrix contains the SoftMax (row-wise) of labels of each node regarding to a target node
    '''
    if topK != -1:
        print("Starting AdaSim* with '{}' on '{}' iterations, top '{}', and C '{}'...".format(graph,iterations,topK,damping_factor)+'\n')
    else:
        print("Starting AdaSim* with '{}' on '{}' iterations, and C '{}'...".format(graph,iterations,damping_factor)+'\n')        

    G = nx.read_edgelist(graph, create_using=nx.DiGraph(), nodetype = int)
    nodes = sorted(G.nodes())       # sorted list of all nodes        
    adj = nx.adjacency_matrix(G,nodelist=nodes, weight=None)      # V*V adjacency matrix
    print("# of nodes in graph: ",len(nodes))
    if topK == -1: ## calculating topK for the input graph
        topK = round(G.number_of_edges()/G.number_of_nodes()) * 8
        print("TopK is calculated and set as '{}' ...".format(topK)+'\n')        
    topK = topK +1 ## a node itself is also considered in the topK list as the most similar one to itself
    degrees = adj.sum(axis=0).T   # V*1 matrix (a column vector of size V)        
    weights = csr_matrix(1/np.log(degrees+math.e))  # keep weights of nodes; V*1 matrix;
    weight_matrix = csr_matrix(adj.multiply(weights)) # V*V matrix; column i have the weight of i's in-neighbors

    print('Iteration 1 ...')
    adamic_scores = weight_matrix + weight_matrix.T + damping_factor * weight_matrix.T * adj
    adamic_scores.setdiag(0) ## $$ corresponding to the ∧ opertaor$$
    adamic_scores = adamic_scores/np.max(adamic_scores)  # min-max normalization        

    result_matrix = 0.5 * adamic_scores
    result_matrix.setdiag(1)
    if iterations == 1:
        if loss=='listMLE_topK':
            top_indices,top_simvals = prData.get_listMLE_topK(result_matrix,topK)
            return top_indices,top_simvals,topK-1

    weight_matrix = normalize(weight_matrix, norm='l1', axis=0) # column normalized weight_matrix    
    for itr in range (2, iterations+1):           
        print("Iteration "+str(itr)+' ...')
        result_matrix.setdiag(0)   ## diagonal values MUST set back to zero
        temp = result_matrix * weight_matrix
        result_matrix =  0.5 * adamic_scores + damping_factor/2.0 * (temp + temp.T)

    if loss=='listMLE_topK':
        result_matrix.setdiag(1) ## set back diagonal values to one
        top_indices,top_simvals = prData.get_listMLE_topK(result_matrix,topK)
        return top_indices,top_simvals, topK-1
                

def compute_only_AdaSim_star (graph='', iterations=0, damping_factor=0.8):
    '''
        Computes AdaSim* measure
    '''
    print("Starting AdaSim* with '{}' on '{}' iterations, and C '{}'...".format(graph,iterations,damping_factor)+'\n')

    G = nx.read_edgelist(graph, create_using=nx.DiGraph(), nodetype = int)
    nodes = sorted(G.nodes())       # sorted list of all nodes
    adj = nx.adjacency_matrix(G,nodelist=nodes, weight=None)      # V*V adjacency matrix
    print("# of nodes in graph: ",len(nodes))
    degrees = adj.sum(axis=0).T   # V*1 matrix (a column vector of size V)
    weights = csr_matrix(1/np.log(degrees+math.e))  # keep weights of nodes; V*1 matrix;
    weight_matrix = csr_matrix(adj.multiply(weights)) # V*V matrix; column i have the weight of i's in-neighbors

    print('Iteration 1 ...')
    adamic_scores = weight_matrix + weight_matrix.T + damping_factor * weight_matrix.T * adj
    adamic_scores.setdiag(0) ## $$ corresponding to the ∧ opertaor$$
    adamic_scores = adamic_scores/np.max(adamic_scores)  # min-max normalization

    result_matrix = 0.5 * adamic_scores
    result_matrix.setdiag(1)
    print (result_matrix.todense())     ## you can write down the result_matrix in a file or process it here

    weight_matrix = normalize(weight_matrix, norm='l1', axis=0) # column normalized weight_matrix
    for itr in range (2, iterations+1):
        print("Iteration "+str(itr)+' ...')
        result_matrix.setdiag(0)   ## diagonal values MUST set back to zero
        temp = result_matrix * weight_matrix
        result_matrix =  0.5 * adamic_scores + damping_factor/2.0 * (temp + temp.T)
        result_matrix.setdiag(1) ## set back diagonal values to one for writing results
        print (result_matrix.todense())     ## you can write down the result_matrix in a file or process it here

