#!-*- coding: utf8 -*-
import utils.variables as maker
from numpy import prod

"""
Pre-Trained weights available in https://drive.google.com/file/d/0B9Ak9mXsx-zQaUtSSmhTS2J6UE0/view?usp=sharing
"""
def create_structure(tf, x):
    """
    Weights
    """
    with tf.variable_scope("structure"):
        conv1 = maker.weight_variable(tf, shape=[11, 11, 3, 96], name="conv1")
        b1 = maker.bias_variable(tf, shape=[96], name="bias1")

        conv2 = maker.weight_variable(tf, shape=[5, 5, 48, 256], name="conv2")
        b2 = maker.bias_variable(tf, shape=[256], name="bias2")

        conv3 = maker.weight_variable(tf, shape=[3, 3, 256, 384], name="conv3")
        b3 = maker.bias_variable(tf, shape=[384], name="bias3")

        conv4 = maker.weight_variable(tf, shape=[3, 3, 192, 384], name="conv4")
        b4 = maker.bias_variable(tf, shape=[384], name="bias4")

        conv5 = maker.weight_variable(tf, shape=[3, 3, 192, 256], name="conv5")
        b5 = maker.bias_variable(tf, shape=[256], name="bias5")


        """
        Architecture
        """
        # x_image = tf.reshape(x, [-])
        h_conv1 = tf.nn.relu(maker.conv2d(tf, x, conv1, strides=[1,4,4,1]) + b1)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(h_conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        m1 = maker.max_pool(tf, lrn1, ksize=[1,3,3,1], padding='VALID')

        input_groups = tf.split(m1, 2, 3)
        kernel_groups = tf.split(conv2, 2, 3)
        output_groups = [maker.conv2d(tf, input_groups[0], kernel_groups[0]), maker.conv2d(tf, input_groups[1], kernel_groups[1])]
        h_conv2 = tf.nn.relu(tf.concat(output_groups, 3) + b2)

        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(h_conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

        m2 = maker.max_pool(tf, lrn2, ksize=[1,3,3,1], padding='VALID')



        h_conv3 = tf.nn.relu(maker.conv2d(tf, m2, conv3) + b3)

        input_groups = tf.split(h_conv3, 2, 3)
        kernel_groups = tf.split(conv4, 2, 3)
        output_groups = [maker.conv2d(tf, input_groups[0], kernel_groups[0]), maker.conv2d(tf, input_groups[1], kernel_groups[1])]
        h_conv4 = tf.nn.relu(tf.concat(output_groups, 3) + b4)
        # h_conv4 = tf.nn.relu(maker.conv2d(tf, h_conv3, conv4) + b4)

        input_groups = tf.split(h_conv4, 2, 3)
        kernel_groups = tf.split(conv5, 2, 3)
        output_groups = [maker.conv2d(tf, input_groups[0], kernel_groups[0]), maker.conv2d(tf, input_groups[1], kernel_groups[1])]
        h_conv5 = tf.nn.relu(tf.concat(output_groups, 3) + b5)
        # h_conv5 = tf.nn.relu(maker.conv2d(tf, h_conv4, conv5) + b5)

        m5 = maker.max_pool(tf, h_conv5, ksize=[1,3,3,1], padding='VALID')
        m5_flat = tf.reshape(m5, [-1, int(prod(m5.get_shape()[1:]))])

        fc1 = maker.weight_variable(tf, shape=[m5_flat.get_shape()[1], 4096], name="fc1")
        bfc1 = maker.bias_variable(tf, shape=[4096], name="bfc1")

        fc2 = maker.weight_variable(tf, shape=[4096, 4096], name="fc2")
        bfc2 = maker.bias_variable(tf, shape=[4096], name="bfc2")

        fc3 = maker.weight_variable(tf, shape=[4096, 1000], name="fc3")
        bfc3 = maker.bias_variable(tf, shape=[1000], name="bfc3")


        h_fc1 = tf.nn.relu(tf.matmul(m5_flat, fc1) + bfc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, fc2) + bfc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, fc3) + bfc3)

        return tf.nn.softmax(h_fc3)
