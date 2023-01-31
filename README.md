# GELTOR: A Graph Embedding Method based on Listwise Learning to Rank**

This repository provides a reference implementation of GELTOR as well as access to the data.

## Installation and usage
GELTOR is a straightforward embedding method implemented by a simple deep neural network consisting of only a projection layer and an output layer.
In order to run GELTOR, the following package are required:
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
A graph must be represented as a text file under the *edge list format* in which, each line corresponds to an edge in the graph, tab is used as the eliminator between two nodes, and the node index is started from 0. 

## Running GELTOR

GELTOR has the following parameters: 
```
--graph: Input graph 
--dataset_name: dataset name 
--result_dir: Destination to save the embedding result, default is "output/" in the root directory 
--dim: The embedding dimension, default is 128 
--topk_mnl: The flag indicating to input topK value manually or to calculate it automatically, default is False 
--topk: Number of nodes in topK, default is -1 
--itr: Number of Iterations to compute AdaSim*, default is 5 
--epc: Number of Epochs for training, default is 300 
--bch_mnl: The flag indicating to input batch size manually or to calculate it automatically, default is False 
--bch: Number of examples in a batch, default is -1 
--lr: Learning rate, default is 0.0025 
--reg: Regularization parameter which is suggested to be 0.001 and 0.0001 with directed and undirected graphs, respectively; default is 0.001 
--early_stop: The flag indicating to stop the training process if the loss stops improving, default is True 
--wait_thr: Number of epochs with no loss improvement after which the training will be stopped, default is 20 
--gpu: The flag indicating to run GELTOR on GPU, default is True
```
### NOTE:
1- By using **topk_mnl** flag, you can input topK value manually (i.e., **topk_mnl=True**) or to calculate it automatically as explained in the paper (i.e., **topk_mnl=False**), default is False. **It is worth to note that topk plays an important role in GELTOR; therefore, in order to obtain
the best effectiveness with any datasets, it is highly recommended to perform an appropriate parameter tuning on topk as explained in the paper, Section 4.3.1.**

2- By using **bch_mnl** flag, you can input the batch size manually (i.e., **bch_mnl=True**) or to calculate it automatically as explained in the paper (i.e., **bch_mnl=False**), default is False; the latter case is recommended.

3- **It is recommended** to set the regularization parameter as 0.001 and 0.0001 with directed and undirected graphs, respectively.

### Sample:
```
python GELTOR.py --graph data/DBLP/DBLP_directed_graph.txt --dataset_name DBLP
```
```
python GELTOR.py --graph data/DBLP/DBLP_directed_graph.txt --dataset_name DBLP --topk_mnl True --topk 50 --bch_mnl True --bch 256
```
## Citation
