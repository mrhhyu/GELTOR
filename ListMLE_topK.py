'''
Created on Jul 25, 2022

Compute the listMLE loss function based on Top k (Statistical Consistency of Top-k Ranking [Xia et al, 2009])

@author: masoud
'''
import tensorflow as tf
from tensorflow.keras.losses import Loss

class ListMLELoss_topK(Loss):
    def __init__(self):
        super(ListMLELoss_topK, self).__init__()   
          
    def call (self,y_true,y_pred):
        '''
            @param y_true: indexes of Top-k nodes 
            @param y_pred: output of the DNN
        '''
        raw_max = tf.reduce_max(input_tensor=y_pred, axis=1, keepdims=True)
        y_pred = y_pred - raw_max   
        sum_all = tf.reduce_sum(input_tensor=tf.exp(y_pred), axis=1, keepdims=True) # summation of exp(x) for all values;
        y_ture_scores = tf.gather(y_pred,y_true,axis=1,batch_dims=1) # Fetch the similarity scores of topK nodes
        cumsum_y_ture_scores = tf.cumsum(tf.exp(y_ture_scores), axis=1, reverse=False, exclusive=True) # cumulative sum for exp of y_ture_scores   
        final_sum = sum_all - cumsum_y_ture_scores
        loss_values = tf.math.log(tf.math.abs(final_sum) + tf.keras.backend.epsilon()) - y_ture_scores
        negative_log_likelihood = tf.reduce_sum(input_tensor=loss_values, axis=1, keepdims=True)
        return negative_log_likelihood
        
