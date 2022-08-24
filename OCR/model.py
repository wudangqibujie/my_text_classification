import tensorflow as tf
import numpy as np


class CRNN_CTC:
    def __init__(self, num_cls, height, width, label_max_length, last_channels):
        self.X = tf.placeholder(tf.float32, [None, height, width, 3])
        self.y = tf.sparse_placeholder(tf.float32)

        self.filters = [1, 64, 128, 128, last_channels]
        self.strides = [1, 2]

        x = self.X
        with tf.variable_scope("cnn"):
            for ix, channels in enumerate(self.filters):

    def max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding="SAME", name="max_pool")

    def leaky_relu(self, x, leakiness=0.8):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    def conv(self, name, x, filters, strides):
        with tf.variable_scope(name):
            kernal = tf.get_variable(name="W",
                                     shape=filters,
                                     dtype=tf.float32,
                                     initializer=tf.global_variables_initializer())
            b = tf.get_variable(name="b",
                                shape=[filters[-1]],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())
            conv_op = tf.nn.conv2d(x, kernal, [1, strides, strides, 1], padding="SAME")
        return tf.nn.bias_add(conv_op, b)

    def batchnorm(self, name, x, is_traing):
        with tf.variable_scope(name):
            x_bn = tf.contrib.layers.batch_norm(
                inputs=x,
                decay=0.9,
                center=True,
                scale=True,
                epsilon=1e-5,
                updates_collections=None,
                is_training=is_traing,
                fused=True,
                data_format='NHWC',
                zero_debias_moving_mean=True,
                scope='BatchNorm'
            )
        return x_bn
