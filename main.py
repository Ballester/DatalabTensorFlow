"""
Definitions
"""
import config.sketch as config
from structures.alexnet import create_structure
from readers.sketch import Dataset
from postprocessing.sketch import post_process

"""
Core
"""
import tensorflow as tf
import numpy as np

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=config.input_size, name="input")
y_ = tf.placeholder(tf.float32, shape=config.output_size, name="expected")

y = create_structure(tf, x)
sess.run(tf.global_variables_initializer())

# try:
post_process(tf, sess)
# except:
    # print("No post processing, skipping")

loss = -tf.reduce_sum(y*tf.log(y_ + 1e-9))
train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)

dataset = Dataset()

"""
Training
"""
for i in range((dataset.get_training_size() * config.epochs)/config.batch_size):
    train_x, train_y = dataset.next_batch(config.batch_size)
    feed_dict = {x: train_x, y_: train_y}
    sess.run(train_step)

"""
Do experiment here:
"""
from utils.sketch.caffe_classes import class_names
# for i in range(dataset.get_test_size()):
# x_test, y_test = dataset.next_test(config.batch_size)
from scipy.misc import imread
from scipy.misc import imresize

x_test = [imresize((imread('/home/ballester/Documents/bathtub.jpg')[:,:,:]).astype(np.float32), (227, 227, 3))]
y_test = [[0.0]*1000]

output = sess.run(y, feed_dict={x: x_test, y_: y_test})

inds = np.argsort(output)[0,:]
# expected_number = output[0].index(1.0)
for i in range(5):
    print class_names[inds[-1-i]], output[0, inds[-1-i]]
