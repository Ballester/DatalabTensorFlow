## Structures
Contains the network's architecture definition. Contains a function **def create_structure(tf, x):** which receives the tf (**import tensorflow as tf**) and the placeholder x (**x = tf.placeholder(tf.float32, shape=config.input_size, name="input")**). This function must the return the last tensor executed by the structure, for example **return tf.nn.softmax(h_fc3)**
