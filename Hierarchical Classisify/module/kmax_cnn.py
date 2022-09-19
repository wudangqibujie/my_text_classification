import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.int32, shape=[None, None])
# print(np.random.randint(0, 120, size=[64, 100]))
embed_table = tf.Variable(tf.random_uniform([3600, 64], -1, 1))
embeded = tf.nn.embedding_lookup(embed_table, x)
parallels = []
kernals = [3, 4, 5]
n_filters = 250


def add_conv1d(x, n_filters, keral_size, strides=1):
    return tf.layers.conv1d(inputs=x,
                            filters=n_filters,
                            kernel_size=keral_size,
                            strides=strides,
                            padding="valid",
                            use_bias=True,
                            activation=tf.nn.relu)


def add_kmax_pooling(x):
    y = tf.transpose(x, [0, 2, 1])
    y = tf.nn.top_k(y, 5, sorted=False).values
    y = tf.transpose(y, [0, 2, 1])
    return tf.reshape(y, [-1, 5, n_filters // len(kernals)])


for k in kernals:
    p = add_conv1d(embeded, n_filters // len(kernals), keral_size=k) #[batch_size, ]
    p = add_kmax_pooling(p)
    parallels.append(p)
parallels = tf.concat(parallels, axis=-1)
parallels = tf.reshape(parallels, [-1, 5 * (len(kernals)*(n_filters//len(kernals)))])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
out = sess.run(parallels, feed_dict={x: np.random.randint(0, 120, size=[64, 100])})
print(out.shape)
# class KmaxCnn:
#     def __init__(self, top_k = 5, n_filters=250, kernals = [3, 4, 5]):

