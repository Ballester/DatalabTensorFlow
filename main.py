#!-*- coding: utf8 -*-
"""
Definitions
"""
import config.apples as config
# from structures.alexnet import create_structure
from structures.alexnet_apples import create_structure
from readers.apples import Dataset
from postprocessing.alexnet_weights_apples import post_process

"""
Core
"""
import tensorflow as tf
import numpy as np
import os

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=config.input_size, name="input")
y_ = tf.placeholder(tf.float32, shape=config.output_size, name="expected")

prob = create_structure(tf, x)

post = post_process(tf)

loss = -tf.reduce_sum(prob*tf.log(y_ + 1e-9))
train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

sess.run(tf.global_variables_initializer())
sess.run(post)

dataset = Dataset()

"""
Summaries
"""
tf.summary.scalar('loss', loss)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join('/tmp', config.log_dir), sess.graph)

"""
Training
"""
for i in range(int((dataset.get_training_size() * config.epochs)/config.batch_size)):
    if (i%10 == 0):
        print("Training %d" % i)
    train_x, train_y = dataset.next_batch(config.batch_size)
    feed_dict = {x: train_x, y_: train_y}
    summary, _ = sess.run([summaries, train_step], feed_dict=feed_dict)
    train_writer.add_summary(summary, i)


"""
Do experiment here:
"""
correct = 0
done = 0
for i in range(0, dataset.get_test_size(), config.batch_size):
    test_x, test_y = dataset.next_test(config.batch_size)
    feed_dict = {x: test_x, y_: [[0.0]*5]*config.batch_size}
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
