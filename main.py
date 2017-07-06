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
import os

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=config.input_size, name="input")
y_ = tf.placeholder(tf.float32, shape=config.output_size, name="expected")

y = create_structure(tf, x)

# try:
post = post_process(tf, sess)
# except:
    # print("No post processing, skipping")

loss = -tf.reduce_sum(y*tf.log(y_ + 1e-9))
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
# for i in range((dataset.get_training_size() * config.epochs)/config.batch_size):
for i in range(0, 1):
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
    pred = sess.run(y, feed_dict=feed_dict)
    for j in range(config.batch_size):
        if np.argmax(test_y[j]) == np.argmax(pred[j]):
            correct += 1
    done += config.batch_size

print("Tests done: " + str(done))
print("Accuracy: " + str(float(correct)/float(dataset.get_test_size())))
