#!-*- coding: utf8 -*-
def weight_variable(tf, shape, name, stddev=0.1):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    # print(name)
    return tf.get_variable(name, shape)

def bias_variable(tf, shape, name):
    # initial = tf.constant(0.01, shape=shape)
    return tf.get_variable(name, shape)

def conv2d(tf, x, W, strides=[1,1,1,1], padding='SAME'):
    return tf.nn.conv2d(x, W, strides, padding)

def max_pool(tf, x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    return tf.nn.max_pool(x, ksize, strides, padding)
