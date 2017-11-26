import tensorflow as tf
import cv2
import numpy as np
import time



# Variables

def weight_init(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_init(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def avg_pool_4x4(x):
    return tf.nn.avg_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')


X = tf.placeholder(tf.float32,[None,640,480,3])
pool1 = avg_pool_4x4(X)
pool2 = avg_pool_4x4(pool1)

WF1 = weight_init([3600,900])
BF1 = bias_init([900])
flatten = tf.reshape(pool2,[-1,3600])
fc1 = tf.nn.relu(tf.matmul(flatten, WF1) + BF1)
fc1_drop = tf.nn.dropout(fc1, 0.75)

WF2 = weight_init([900, 3])
BF2 = bias_init([3])
result = tf.nn.softmax(tf.matmul(fc1_drop, WF2) + BF2)


# Now evaluation
Y = tf.placeholder(tf.float32,[None,3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(result), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
comp_pred = tf.equal(tf.arg_max(result,1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))

# Create Saver
param_list = [WF1,WF2,BF1,BF2]
saver = tf.train.Saver(param_list)

# Now train it!

