import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.float32, shape=[None, 100, 128, 1])
w = tf.Variable(tf.truncated_normal([1, 128, 1, 64], stddev=0.1))
b = tf.Variable(tf.truncated_normal([64], stddev=0.01))
conved = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME") + b)
pooled = tf.nn.max_pool(conved, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
out = sess.run(pooled, feed_dict={x: np.random.random((64, 100, 128, 1))})
print(out.shape)