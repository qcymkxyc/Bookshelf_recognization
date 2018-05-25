#coding=utf8

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import metrics
from src import util 


def _input_train_fn(features,labels,num_epoch,batch_size = None):
    """
            训练的输入函数
            Args:
                features : x
                labels :  y
                num_epoch :  轮数
                batch_size : batch
            Return:
                x和y的字典形式 
    """

    class_label = util.map_label(labels)
    if isinstance(features,pd.DataFrame):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    (x.to_dict("list"),y)
                                    )
    if isinstance(features, np.ndarray):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    ({"x" : features}, class_label)
                                    )
    
    
    dataset = dataset.shuffle(50000).repeat(num_epoch).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def _input_eval_fn(features,labels,batch_size = None):
    """
        模型评价的输入函数
        
    """

    class_label = util.map_label(labels)
    if isinstance(features,pd.DataFrame):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    (x.to_dict("list"),y)
                                    )
    if isinstance(features, np.ndarray):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    ({"x" : features}, class_label)
                                    )
    
    assert batch_size is not None,"batch_size 不能为空"
    
    dataset = dataset.repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def _input_predict_fn(features,batch_size):
    """
        模型评价的输入函数
        
    """

    if isinstance(features,pd.DataFrame):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    (x.to_dict("list"))
                                    )
    if isinstance(features, np.ndarray):
        dataset = tf.data.Dataset.from_tensor_slices(
                                    ({"x" : features})
                                    )
    
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

class BaseEstimator(object):
    """
                分类器的基类
    """
    
    def __init__(self,                 
                 n_inputs,
                 hidden_units,
                 learning_rate = 0.01,
                 model_dir = "./logs/",
                 n_classes = 2,
                 optimizer = tf.train.AdamOptimizer,
                 config = None):
        """ 
            Args:
                n_inputs : 输入的维度
                hidden_units  :   隐藏层的神经元数
                learning_rate : 优化方法的学习率 
                model_dir:  模型存储的路径
                n_classes:  分类的类别数
                optimizer:  优化器
                config:  关于运行时的配置，如果为None，即为默认的RunConfig
        """
        
        self._n_inputs = n_inputs
        self._hidden_units = hidden_units
        self._learning_rate = learning_rate
        self._n_classes = n_classes
        self._optimizer = optimizer


        self._estimator = tf.estimator.Estimator(
                                config = config,
                                model_dir = model_dir,
                                model_fn = self._model_fn
        )
        
        self._train_spec = None
        self._eval_spec = None
        
    def _model_fn(self,features,labels,mode):
        """
                        模型的主体部分
            Args: 
                features: 
                labels:
                mode:  执行的模式
        """
        metrics.add_learning_rate(self._learning_rate)
        
    def _hidden_layer_builder(self, net,mode):
        """
                        构建隐藏层
        """
        pass;
    
    def _cost_builer(self,logits,labels):
        """
                        构建cost
        """
        pass
        
    def _train_op_builder(self,loss):
        pass
        pass;
    
    def train(self,features,labels,num_epoch,batch_size,max_steps):
        self._estimator.train(
                        input_fn = lambda : _input_train_fn(features, labels, num_epoch, batch_size)
                        )
        

    def evaluate(self,features,labels,batch_size):
        self._estimator.evaluate(
            input_fn = lambda : _input_eval_fn(features, labels, batch_size)
        )
        
        
    def predict(self,features,batch_size):
        pred_y = self._estimator.predict(
             input_fn = lambda : _input_predict_fn(features,batch_size)
        )
        return pred_y
    
    
    def train_and_eval(self,x_train, x_test, y_train, y_test,max_steps,batch_size = 1000):
        """
            
        """
#         x_train, x_test, y_train, y_test = train_test_split(
#             feature, labels, test_size = test_size)

        self._train_spec = tf.estimator.TrainSpec(
                    input_fn= lambda : _input_train_fn(
                                        features = x_train,
                                        labels = y_train,
                                        num_epoch = None,
                                        batch_size = batch_size
                ),
            max_steps = max_steps
            )
        
        self._eval_spec = tf.estimator.EvalSpec(
                        lambda : _input_eval_fn(
                                            features = x_test,
                                            labels = y_test,
                                            batch_size = batch_size
                                        )
            )
        
        tf.estimator.train_and_evaluate(self._estimator,self._train_spec,self._eval_spec)
        
        
        