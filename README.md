# GELTOR: A Graph Embedding Method based on Listwise Learning to Rank

This repository provides a reference implementation of GELTOR as well as access to the data.

## Installation and usage
GELTOR is a similarity-based embedding method that applies the listwise learning-to-rank to graph embedding. It is straightforward embedding method implemented by a simple deep neural network consisting of only a projection layer and an output layer.
In order to run GELTOR, the following packages are required:
```
Python       >= 3.8
tensorflow   >= 2.2
networkx     >=2.6.*
numpy        >=1.21.*
scipy        >=1.7.*
scikit-learn >=1.0.*
```

GELTOR can be run directly from the command line or migrated to your favorite IDE.
## Graph file format
A graph must be represented as a text file under the *edge list format* in which, each line corresponds to an edge in the graph, tab is used as the separator of the two nodes, and the node index is started from 0. 

## Running GELTOR

GELTOR has the following parameters: 
```
--graph: Input graph

--dataset_name: dataset name

--result_dir: Destination to save the embedding result, default is "output/" in the root directory

--dim: The embedding dimension, default is 128

--topk: Number of nodes in topK; default is -1 (i.e., topK is automatically computed as explained in the paper), otherwise set the value manually

--itr: Number of Iterations to compute AdaSim*, default is 5

--epc: Number of Epochs for training, default is 300

--bch: batch size, default is -1 (i.e., it is automatically computed as explained in the paper), otherwise set the value manually

--lr: Learning rate, default is 0.0025

--reg: Regularization parameter which is suggested to be 0.001 and 0.0001 with directed and undirected graphs, respectively; default is 0.001

--early_stop: The flag indicating to stop the training process if the loss stops improving, default is True

--wait_thr: Number of epochs with no loss improvement after which the training will be stopped, default is 20

--gpu: The flag indicating to run GELTOR on GPU, default is True
```

### NOTE:
1- It is worth to note that **topk** plays an important role in GELTOR; therefore, in order to obtain the best effectiveness with any datasets, it is recommended to perform an appropriate parameter tuning on topk as explained in the paper, Section 4.3.1.

2- **It is recommended** to set the regularization parameter as 0.001 and 0.0001 with directed and undirected graphs, respectively.

### Sample:
1) **topk** and **bch** parameters are set _automatically_

```
python GELTOR.py --graph data/DBLP/DBLP_directed_graph.txt --dataset_name DBLP
```

2) **topk** and **bch** parameters are set _manually_

```
python GELTOR.py --graph data/DBLP/DBLP_directed_graph.txt --dataset_name DBLP --topk 50 --bch 256
```
## Citation:
> Masoud Reyhani Hamedani, Jin-Su Ryu, and Sang-Wook Kim. 2023. GELTOR: A Graph Embedding Method based on Listwise Learning to Rank. In Proceedings of the ACM Web Conference 2023 (WWW ’23), April 30-May 4, 2023, Austin, TX, USA, pages: 6-16. https://doi.org/10.1145/3543507.3583193

