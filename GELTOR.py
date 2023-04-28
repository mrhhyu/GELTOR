'''
Created on August 12, 2022

@note Implementation of our proposed method with maximum a posterior (MAP) estimator by considering the topK nodes

@author: masoud
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as tkb
from argparse import ArgumentParser
from AdaSim_star_SE import compute_AdaSim_star
import time
from ListMLE_topK import ListMLELoss_topK
import os
import math

class CustomCallback_verbose_check(tf.keras.callbacks.Callback):
    '''
        1- Representing custom verbose information
        2- checking loss
    '''
    def on_epoch_end(self, epoch, logs=None):
        ## checking loss
        print("Epoch: {} .... lr: {}; loss: {}".format(epoch+1,round(float(tkb.get_value(self.model.optimizer.learning_rate)),5),round(logs['loss'],3)))
    def on_train_end(self, logs=None):
        print("Stop training; .... Final loss: {}".format(round(logs['loss'],3)))

def get_model(dim,out_len, learning_rate, reg_rate):
    model = tf.keras.Sequential()
    model.add(layers.Dense(dim, activation='linear', input_shape=(out_len,), name='layer_0'))
    model.add(layers.Dense(out_len,activation='relu', kernel_regularizer=regularizers.L2(reg_rate), bias_regularizer=regularizers.L2(reg_rate), name='layer_1'))
    #print(model.summary())
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate), loss = ListMLELoss_topK(), run_eagerly = True) # run_eagerly=True --> to enable getting Tensor values via .numpy()
    return model

def LTRG(args):
    print()
    if args.topk_mnl and args.topk <= 0:
        print('ERROR: topK is not valid ... please input the topK value!')
        return
    if args.bch_mnl and args.bch <= 0:
        print('ERROR: batch size is not valid ... please input the batch size!')
        return
    if not os.path.exists(args.graph):
        print('ERROR: graph is invalid ...!')
        return
    if args.dataset_name=='':
        print('ERROR: dataset name is invalid ...!')
        return
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## disabling GPU

    print('==================================================================== GELTOR ARGUMENTS ====================================================================')
    print(args,'\n')

    print('================================================================= Similarity Computation =================================================================')
    if args.topk_mnl:
        top_indices,top_simvals,args.topk = compute_AdaSim_star(graph=args.graph, iterations=args.itr, damping_factor=0.4, topK=args.topk, loss='listMLE_topK')#[0]
    else:
        top_indices,top_simvals,args.topk = compute_AdaSim_star(graph=args.graph, iterations=args.itr, damping_factor=0.4, topK=-1, loss='listMLE_topK')#[0]

    print('===================================================================== Model Training ======================================================================')
    if not args.bch_mnl: # calculating batch size for the input graph
        args.bch = pow(2, round(math.log2(len(top_indices)*0.05)))
    info = args.result_dir+args.dataset_name+'_GELTOR_IT'+str(args.itr)+'_Reg'+str(args.reg).split('.')[1]+'_dim'+str(args.dim)+'_bch'+str(args.bch)+'_Top'+str(args.topk)
    tf_input = tf.eye(len(top_indices), dtype='int32') # on-hot vectors as input
    model = get_model(int(args.dim/2), len(top_indices), args.lr, args.reg)
    if args.early_stop: ## apply early stopping
        callback_EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.wait_thr, mode='min', restore_best_weights=True) ## defines a callback for early stop
        model.fit(x=tf_input,y=top_indices, epochs=args.epc, batch_size=args.bch,callbacks=[CustomCallback_verbose_check(),callback_EarlyStopping],verbose=0)
    else:
        model.fit(x=tf_input,y=top_indices, epochs=args.epc, batch_size=args.bch,callbacks=[CustomCallback_verbose_check()],verbose=0)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    emb_0 = model.get_layer('layer_0').weights[0][:].numpy()
    emb_1 = model.get_layer('layer_1').weights[0][:].numpy().T
    emb_file = open(info+'.emb','w')
    emb_file.write(str(len(emb_0))+'\t'+str(args.dim)+'\n')
    for row in range(0,len(emb_0)):
        emb_file.write(str(row))
        emb_val = ''
        for col in range(0, int(args.dim/2)):
            emb_val = emb_val + '\t' + str(emb_0[row][col])
        for col in range(0, int(args.dim/2)):
            emb_val = emb_val + '\t' + str(emb_1[row][col])
        emb_file.write(emb_val+'\n')
    emb_file.close()
    print('The embedding result is written in the file ....')

def parse_args(graph='',dataset_name='',result_dir='output/', dimension=128, topk_mnl=False, topK=-1, iterations=5, epochs=300, bch_mnl=False, batch_size=-1, learning_rate=0.0025, reg_rate=0.001, early_stop=True, wait_thr=20, gpu_on=True):

    parser = ArgumentParser(description="Run GELTOR, A Graph Embedding Method based on Learning to Rank.")
    parser.add_argument('--graph', nargs='?', default=graph, help='Input graph')
    parser.add_argument('--dataset_name', nargs='?', default=dataset_name, help='dataset name')
    parser.add_argument('--result_dir', nargs='?', default=result_dir, help='Destination to save the embedding result, default is "output/" in the root directory')
    parser.add_argument('--dim', type=int, default=dimension, help='The embedding dimension, default is 128')
    parser.add_argument('--topk_mnl', type=bool, default=topk_mnl, help='The flag indicating to input topK value manually or to calculate it automatically, default is False')
    parser.add_argument('--topk', type=int, default=topK, help='Number of nodes in topK, default is -1')
    parser.add_argument('--itr', type=int, default=iterations, help='Number of Iterations to compute AdaSim*, default is 5')
    parser.add_argument('--epc', type=int, default=epochs, help='Number of Epochs for training, default is 300')
    parser.add_argument('--bch_mnl', type=bool, default=topk_mnl, help='The flag indicating to input batch size manually or to calculate it automatically, default is False')
    parser.add_argument('--bch', type=int, default=batch_size, help='Number of examples in a batch, default is -1')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate, default is 0.0025')
    parser.add_argument('--reg', type=float, default=reg_rate, help='Regularization parameter which is suggested to be 0.001 and 0.0001 with directed and undirected graphs, respectively; default is 0.001')
    parser.add_argument('--early_stop', type=bool, default=early_stop, help='The flag indicating to stop the training process if the loss stops improving, default is True')
    parser.add_argument('--wait_thr', type=int, default=wait_thr, help='Number of epochs with no loss improvement after which the training will be stopped, default is 20')
    parser.add_argument('--gpu', type=bool, default=gpu_on, help='The flag indicating to run GELTOR on GPU, default is True')


    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    LTRG(args)

    '''


    args = parse_args(graph='/home/masoud/backup_1/data/feature_learning/email_EU/dataset/train_test/email_EU_directed_graph.txt',
                  dataset_name='email_EU',
                  result_dir='result_test/',
                  dimension=128,
                  topK=20,
                  iterations=6,
                  epochs=10,
                  batch_size=64,
                  learning_rate=0.0025,
                  reg_rate= 0.001,
                  early_stop=False,
                  wait_thr=20,
                  gpu_on=True
                  )
    LTRG(args)


    '''

