"""
Definitions
"""
import config.apples as config
# from structures.alexnet import create_structure
from structures.alexnet_apples import create_structure
from readers.apples import Dataset
from postprocessing.alexnet_weights import post_process

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
    if (i%10 == 0):
        print("Training %d" % i)
    train_x, train_y = dataset.next_batch(config.batch_size)
    feed_dict = {x: train_x, y_: train_y}
    sess.run(train_step, feed_dict=feed_dict)

"""
Do experiment here:
"""
correct = 0
for i in range((dataset.get_test_size())):
    test_x, test_y = dataset.next_test(config.batch_size)
    feed_dict = {x: test_x, y_: [[0.0]*5]*config.batch_size}
    pred = sess.run(y, feed_dict=feed_dict)
    for j in range(config.batch_size):
        if np.argmax(test_y) == np.argmax(pred):
            correct += 1

print("Test set size: " + str(dataset.get_test_size()))
print("Accuracy: " + str(float(correct)/float(dataset.get_test_size())))
