#!-*- conding: utf8 -*-
from numpy import concatenate, load, argsort
def post_process(tf, args={}):
    net_data = load("data/bvlc_alexnet.npy",encoding='latin1').item()
    ops = []
    with tf.variable_scope("structure", reuse=True):
        conv1 = tf.get_variable("conv1")
        b1 = tf.get_variable("bias1")

        conv2 = tf.get_variable("conv2")
        b2 = tf.get_variable("bias2")

        conv3 = tf.get_variable("conv3")
        b3 = tf.get_variable("bias3")

        conv4 = tf.get_variable("conv4")
        b4 = tf.get_variable("bias4")

        conv5 = tf.get_variable("conv5")
        b5 = tf.get_variable("bias5")

        fc1 = tf.get_variable("fc1")
        bfc1 = tf.get_variable("bfc1")

        fc2 = tf.get_variable("fc2")
        bfc2 = tf.get_variable("bfc2")

        fc3 = tf.get_variable("fc3")
        bfc3 = tf.get_variable("bfc3")

        ops.append(tf.assign(conv1, net_data["conv1"][0]))
        ops.append(tf.assign(b1, net_data["conv1"][1]))

        ops.append(tf.assign(conv2, net_data["conv2"][0]))
        ops.append(tf.assign(b2, net_data["conv2"][1]))

        ops.append(tf.assign(conv3, net_data["conv3"][0]))
        ops.append(tf.assign(b3, net_data["conv3"][1]))

        ops.append(tf.assign(conv4, net_data["conv4"][0]))
        ops.append(tf.assign(b4, net_data["conv4"][1]))

        ops.append(tf.assign(conv5, net_data["conv5"][0]))
        ops.append(tf.assign(b5, net_data["conv5"][1]))

        # ops.append(tf.assign(fc1, net_data["fc6"][0]))
        # ops.append(tf.assign(bfc1, net_data["fc6"][1]))
        #
        # ops.append(tf.assign(fc2, net_data["fc7"][0]))
        # ops.append(tf.assign(bfc2, net_data["fc7"][1]))
        #
        # ops.append(tf.assign(fc3, net_data["fc8"][0]))
        # ops.append(tf.assign(bfc3, net_data["fc8"][1]))

        # print(ops)
        return ops
        # print(b1.eval(session=sess))
        # print(net_data["conv1"][1])
