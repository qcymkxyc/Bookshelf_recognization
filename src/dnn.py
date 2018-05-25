#coding=utf8

import tensorflow as tf
import base_estimator
from base_estimator import BaseEstimator
import metrics
import util

class DNNClassifier(BaseEstimator):
    """
        DNN分类器
    """
    def __init__(self,
                 n_inputs,
                 hidden_units,
                 learning_rate = 0.01,
                 model_dir = "./logs/",
                 n_classes = 2,
                 optimizer = tf.train.AdamOptimizer,
                 config = None
                 ):
        """ 
            Args:
                model_dir:  模型存储的路径
                n_classes:  分类的类别数
                hidden_units:   隐藏层的神经元数
                feature_columns:   属性列
                lstm_index:  LSTM层在隐藏层的第几层，默认是第0层
                optimizer:  优化器
                config:  关于运行时的配置，如果为None，即为默认的RunConfig
        """
        
        super(DNNClassifier,self).__init__(
                n_inputs = n_inputs,
                hidden_units = hidden_units,
                learning_rate = learning_rate,
                model_dir = model_dir,
                n_classes = n_classes,
                optimizer = optimizer,
                config = config
            )
        
    def _model_fn(self,features,labels,mode):
        """
                    模型的主体部分
                    Args:
                        features: 
                        labels:
                        mode:  执行的模式
        """
        super(DNNClassifier,self)._model_fn(features,labels,mode)
        feature_columns = [tf.feature_column.numeric_column(
                                                        key = "x",
                                                        shape = self._n_inputs
        )]
        
        net = tf.feature_column.input_layer(features,feature_columns)
        
        for i,units in enumerate(self._hidden_units):
            with tf.name_scope("hidden_layer_{0}".format(i)):
                net = tf.layers.dense(
                                inputs = net,
                                units = units,
                                activation = tf.nn.relu
                )
                tf.summary.histogram(str(units),net)
        
        logits = tf.layers.dense(inputs = net,units = self._n_classes,activation = None)
        
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
#         loss = util.weight_loss(logits = logits,labels = labels)
        
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
    
       
