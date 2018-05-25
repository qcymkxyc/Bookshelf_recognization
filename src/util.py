#coding=utf8

import numpy as np
import base_estimator
import tensorflow as tf
import pandas as pd
from tensorflow.core.framework.cost_graph_pb2 import _COSTGRAPHDEF_NODE_OUTPUTINFO

#标签的list
label_list = ["D00","DBB","DB0","D05","DHT","D10"]

    #代价矩阵
cost_matrics = np.array([
    [0,1,1,1,1,1],
    [5,0,1,1,1,1],
    [5,1,0,1,1,1],
    [5,1,1,0,1,1],
    [5,1,1,1,0,1],
    [5,1,1,1,1,0]
],dtype = np.float32)
    
def get_cost_matrics():
    """
                返回代价矩阵
    """
    
    cols_tuple = list(zip(["Prediction Values"] * len(label_list),label_list))
    indexs_tuple = list(zip(["True Valus"] * len(label_list),label_list))
    
    indexs = pd.MultiIndex.from_tuples(list(indexs_tuple))
    cols = pd.MultiIndex.from_tuples(list(cols_tuple))
    
    cost_df = pd.DataFrame(cost_matrics,index = indexs,columns = cols)
    
    return cost_df
    
    
def gradient_ckeck():
    """
                梯度检验
        Args:
            inputs
    """
    pass;


def build_input_layer(n_inputs):
        feature_columns = [tf.feature_column.numeric_column(
                                                        key = "x",
                                                        shape = n_inputs
    )]


def weight_loss(logits,labels):
    """
                加入权重的loss
    """
    predictions = tf.nn.softmax(logits)
    prediction_labels = tf.argmax(predictions, 1)
    
    one_hot_labels = tf.one_hot(
                        indices = labels, 
                        depth = len(label_list),
                        on_value = 1.,
                        off_value = 0., 
                        name = "one_hot_labels",
                        dtype = tf.float32)
    #获取对应权重
    loss_weights = tf.matmul(one_hot_labels,cost_matrics)
#     print(loss_weights)
#     print(predictions)
#     print(one_hot_labels)
#     
#     print(tf.log)
    loss = - tf.reduce_mean(tf.log(predictions + 10e-7) * one_hot_labels * loss_weights
                            ,name = "loss")
    
    return loss
    
def map_label(label_str):
    """
        将标签的str形式转换成int形式
    """
    if label_str is None:
        return 
    f = lambda x : label_list.index(x) if isinstance(x, str) else x
    class_label = list(map(f,label_str))
    class_label = np.array(class_label)
        
    return class_label
    