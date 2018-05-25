#coding=utf8

from lstm import LSTMClassifier
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import estimator
import base_estimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from base_estimator import BaseEstimator
import metrics   

class BLSTMClassifier(LSTMClassifier):
    
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

        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                                    num_units = units,
                                    forget_bias = 1.0,
                                    state_is_tuple = True)

        fw_init_state = fw_lstm_cell.zero_state(
                                    batch_size = self._batch_size,
                                    dtype = tf.float32
                                            )
        
        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                                    num_units = units,
                                    forget_bias = 1.0,
                                    state_is_tuple = True)
        
        bw_init_state = bw_lstm_cell.zero_state(
                                    batch_size = self._batch_size,
                                    dtype = tf.float32
                                    )

        outputs,states = tf.nn.bidirectional_dynamic_rnn(
                                                cell_fw = fw_lstm_cell,
                                                cell_bw = bw_lstm_cell,
                                                inputs = X_in,
                                                initial_state_fw = fw_init_state,
                                                initial_state_bw = bw_init_state,
                                                )
        state_fw, state_bw = states
        combine_state = tf.concat([state_fw[1],state_bw[1]],axis = 1)
        return combine_state
    
    