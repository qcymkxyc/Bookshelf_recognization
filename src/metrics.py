#coding=utf8
"""
   评价标准文件 
"""

import tensorflow as tf
from tensorboard import summary as summary_lib

def add_fbeta_score(y_pred,y,beta):
    pass


def add_loss(loss):
    """
                添加loss
    """
    tf.summary.scalar("loss",loss)
    

def add_accuracy(logits,labels,metrics = None):
    """
                添加精度
    """
    if metrics is None:
        metrics = {}
    predict_classes = tf.argmax(logits,1)
    accuracy = tf.metrics.accuracy(
                                labels = labels,
                                predictions = predict_classes,
                                name = "acc_op"
    )
    metrics["accuracy"] = accuracy
    tf.summary.scalar(name = "accuracy",tensor = accuracy[1])
    
    return metrics
    
def add_learning_rate(learning_rate):
    """
                添加学习率
    """
    tf.summary.scalar("learning_rate",learning_rate)
    
    
def add_keep_prob(keep_prob):
    """
                添加dropout的keep_prob
    """
    if keep_prob is not None:
        tf.summary.scalar("keep_prob",keep_prob)
        
# def add_pr(pred_y,y):
#     summary_lib.pr_curve_streaming_op(
#                                 tag = "PR_CURVE",
#                                 predictions = pred_y,
#                                 labels = y
# #                                  num_thresholds = 10
#         )