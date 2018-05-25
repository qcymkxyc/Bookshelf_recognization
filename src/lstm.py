#coding=utf8

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import estimator
import base_estimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from base_estimator import BaseEstimator
import metrics   
    
class LSTMClassifier(BaseEstimator):
    """
        LSTM分类器
    """
    def __init__(self,
                 n_inputs,
                 model_dir,
                 n_classes,
                 hidden_units,
                 n_steps = 1,
                 lstm_index = 0,
                 learning_rate = 0.01,
                 optimizer = tf.train.AdamOptimizer,
                 config = None,
                 dropout = 0.2,
                 teacher_force = False,
                 batch_size = 1000,
                 epoch = 10
    ):
        """
            Args:
                model_dir:  模型存储的路径
                n_classes:  分类的类别数
                hidden_units:   隐藏层的神经元数
                feature_columns:   属性列
                n_steps:   RNN的阶数
                lstm_index:  LSTM层在隐藏层的第几层，默认是第0层
                optimizer:  优化器
                config:  关于运行时的配置，如果为None，即为默认的RunConfig
                dropout: 是否采用dropout，默认为None，表示不采用，如果采用，
                该列需填入keep_prob的值，采用 的dropout使用Inverse Dropout方式
                batch_size :  
                epoch :  
        """
        self._lstm_index = lstm_index
        self._dropout = dropout
        self._batch_size = batch_size
        self._epoch = epoch
        self._n_steps = n_steps

        super(LSTMClassifier,self).__init__(
                n_inputs = n_inputs,
                hidden_units = hidden_units,
                learning_rate = learning_rate,
                model_dir = model_dir,
                n_classes = n_classes,
                optimizer = optimizer,
                config = config
            )
        

    def _model_fn(self, features, labels, mode):
        """
                        构建模型的主体部分
            Args:
                features : 属性
                labels :   标签
                mode :   模式
        """
        super(LSTMClassifier,self)._model_fn(features,labels,mode)
        feature_columns = [tf.feature_column.numeric_column(
                                            key = "x",
                                            shape = (self._n_steps,self._n_inputs)
        )] 
        
        
        #创建输入层
        net = tf.feature_column.input_layer(features,feature_columns)

        #建立隐藏层
        net = self._hidden_layer_builder(net,mode)
        logits = tf.layers.dense(net,self._n_classes,activation = None)
        
        tf.summary.histogram("logits",logits)
        

        predict_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                        'class_ids': predict_classes[:, tf.newaxis],
                        "probabilities" : tf.nn.softmax(logits,),
                        'logits': logits,
                    }
            return tf.estimator.EstimatorSpec(mode, predictions = predictions)
        
        loss = tf.losses.sparse_softmax_cross_entropy(labels = labels,logits = logits)
        
        metrics_dict = metrics.add_accuracy(logits,labels)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                                mode = mode,
                                loss = loss,
                                eval_metric_ops = metrics_dict
            )
            
            
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = self._optimizer(learning_rate = self._learning_rate);
        train_op = optimizer.minimize(loss,global_step = tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

    def _hidden_layer_builder(self, net,mode):
        """
                        添加隐藏层,该方法是用迭代方式构建
            Args:
                net : 输入层
                _layer_index : 表示构建到第几层，外部调用不使用该参数
            Return:
                            构建好的网络
            Raises:
                TypeError : 如果隐藏层的数目不是list类型
        """
        if not isinstance(self._hidden_units,list):
            raise TypeError(r"隐藏层必须为list类型")
        
        net = tf.reshape(net, [-1,self._n_inputs])
        for i,value in enumerate(self._hidden_units):
            net = tf.layers.dense(
                        inputs = net,
                        units = value,
                        activation = tf.nn.relu)
            
            #建立lstm层
            if self._lstm_index == i:
                states = self._lstm_layer_builder(
                        net, value)
                net = states
                continue;            
        
            if self._dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                tf.layers.dropout(net,rate = self._dropout,training = True)
            tf.summary.histogram(name="layer_{}".format(i), values = net)
        return net

    def _lstm_layer_builder(self, net, units):
        """
                        建立lstm层
            Args:
                net : 输入层
                units ： lstm层的states的维数
            Return :
                (outputs,states):
                                返回的outputs形式为（batch_size,max_time,ouput_size）
                states的形式为一个Tuple，states[0]为(batch_size,output_size)
                states[1](batch_size,states_size)
            Raise :
                IndexError : 如果lstm层数的索引在隐藏层之外
        """
        if self._lstm_index >= len(self._hidden_units):
            raise IndexError("LSTM层的索引必须小于隐藏层的层数")
        
        X_in = tf.reshape(net, [-1, self._n_steps, net.shape[1]])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                                    num_units = units,
                                    forget_bias = 1.0,
                                    state_is_tuple = True)

        _init_state = lstm_cell.zero_state(
                                    batch_size = self._batch_size,
                                    dtype = tf.float32
                                            )
        #返回的outputs形式为（batch_size,max_time,ouput_size）
        #states的形式为(batch_size,states)
        outputs,states = tf.nn.dynamic_rnn(
                                    cell = lstm_cell ,
                                    inputs = X_in,
                                    initial_state = _init_state,
                                    time_major = False)

#         return outputs,states
        return states[1]
    
    def train_and_eval(self,x_train, x_test, y_train, y_test,max_steps,batch_size = 1000):
        """
            
        """
        x_train_in = x_train.reshape(-1,self._n_steps,self._n_inputs)
        x_test_in = x_test.reshape(-1,self._n_steps,self._n_inputs)
        
        super(LSTMClassifier,self).train_and_eval(x_train_in,x_test_in,y_train,y_test,
                                                         max_steps,self._batch_size)

