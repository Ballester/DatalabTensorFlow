#!-*- coding: utf8 -*-
"""
Definitions
"""
import config.apples as config
# from structures.alexnet import create_structure
from structures.alexnet_apples import create_structure
from readers.apples import Dataset
from postprocessing.alexnet_weights import post_process

#from vis.visualization import visualize_cam, overlay
#from vis.utils import utils
#from keras import activations

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import utils_visualize as utils

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


"""
Core
"""
import tensorflow as tf
import numpy as np
import os
from keras import backend as K

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
#plt.use('Agg')

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=config.input_size, name="input")
y_ = tf.placeholder(tf.float32, shape=config.output_size, name="expected")

prob, pool = create_structure(tf, x)
post = post_process(tf)

loss = -tf.reduce_sum(prob*tf.log(y_ + 1e-9))
#model.compile('adam', 'mean_squared_error')

train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

target_conv_layer = pool
cost = tf.reduce_sum((prob - y_) ** 2)
target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
gb_grad = tf.gradients(cost, x)[0]

target_conv_layer_grad_norm = tf.div(target_conv_layer_grad, tf.sqrt(tf.reduce_mean(tf.square(target_conv_layer_grad))) + tf.constant(1e-5))

sess.run(tf.global_variables_initializer())
sess.run(post)

args = {'input_size': config.input_size}
dataset = Dataset(args)

"""
Summaries
"""
tf.summary.scalar('loss', loss)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join('/tmp', config.log_dir), sess.graph)

#correct = 0
#done = 0
#for i in range(0, dataset.get_test_size(), config.batch_size):
#    test_x, test_y = dataset.next_test(config.batch_size)
#    feed_dict = {x: test_x, y_: [[0.0]*5]*config.batch_size, K.learning_phase(): 0}
#    pred = sess.run(prob, feed_dict=feed_dict)
#    for j in range(config.batch_size):
#        if np.argmax(test_y[j]) == np.argmax(pred[j]):
#            correct += 1
#    done += config.batch_size

#print("Tests done: " + str(done))
#print("Accuracy: " + str(float(correct)/float(dataset.get_test_size())))

"""
Training
"""
eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

        cont = 0
        for i in range(int((dataset.get_training_size() * config.epochs)/config.batch_size)):
            if (i%200 == 0):
                print("Training %d" % i)
                correct = 0
                done = 0
                cm = np.zeros((5,5))
                #print(dataset.get_test_size())
                for j in range(0, dataset.get_test_size(), config.batch_size):
                    #print(j)
                    test_x, test_y = dataset.next_test(config.batch_size)
                    feed_dict = {x: test_x, y_: [[0.0]*5]*config.batch_size, K.learning_phase(): 0}
                    pred = sess.run(prob, feed_dict=feed_dict)
                    #print ("Pred and Test")
                    #print(pred)
                    #print(np.array(test_y))
                    for k in range(config.batch_size):
                        #print(k)
                        cont += 1
                        if np.argmax(test_y[k]) == np.argmax(pred[k]):
                            correct += 1
                        cm[np.argmax(test_y[k])][np.argmax(pred[k])] += 1

                    done += config.batch_size
                print(cm)
                heats_im, heats_gt = dataset.get_examples_of_each_class()
                #print heats_im[0]
                #print(heats_gt)
                gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad_norm], feed_dict={x: heats_im, y_: heats_gt})

                for l in range(0, config.batch_size):
                    utils.visualize(heats_im[l], target_conv_layer_value[l], target_conv_layer_grad_value[l], gb_grad_value[l], i+l)
                    #heatmap = visualize_cam(model, -1, filter_indices=None, seed_input=heats_im[l], backprop_modifier='guided')
                    #plt.imsave('heat_' + str(l) + '_'  + str(i) + '.png', overlay(heats_im[l], heatmap))

                print("Tests done: " + str(done))
                with open("alexnet.txt", "a") as fid:
                    fid.write(str(float(correct)/float(dataset.get_test_size())) + "\n")
                print("Accuracy: " + str(float(correct)/float(dataset.get_test_size())))
                print(cont)
            train_x, train_y = dataset.next_batch(config.batch_size)
            feed_dict = {x: train_x, y_: train_y, K.learning_phase(): 1}
            summary, _ = sess.run([summaries, train_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, i)


"""
Do experiment here:
"""
correct = 0
done = 0
for i in range(0, dataset.get_test_size(), config.batch_size):
    test_x, test_y = dataset.next_test(config.batch_size)
    feed_dict = {x: test_x, y_: [[0.0]*5]*config.batch_size, K.learning_phase(): 0}
    pred = sess.run(prob, feed_dict=feed_dict)
    for j in range(config.batch_size):
        if np.argmax(test_y[j]) == np.argmax(pred[j]):
            correct += 1
    done += config.batch_size

print("Tests done: " + str(done))
print("Accuracy: " + str(float(correct)/float(dataset.get_test_size())))

"""
Example of single feedforward
"""
# from scipy.misc import imread
# from scipy.misc import imresize
# from utils.sketch.caffe_classes import class_names
# x_test = [imresize((imread('/home/ballester/Documents/bathtub.jpg')[:,:,:]).astype(np.float32), (227, 227, 3))]
# y_test = [[0.0]*1000]
#
# output = sess.run(prob, feed_dict={x: x_test, y_: y_test})
# inds = np.argsort(output)[0,:]
# # expected_number = output[0].index(1.0)
# for i in range(5):
#   print class_names[inds[-1-i]], output[0, inds[-1-i]]
