# import random
# import re
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
# sen = "弱小的我也有大梦想"
# tokens = tokenizer.tokenize(sen)
# print(tokens)
import tensorflow as tf
import numpy as np
# a = np.ones(shape=(10, 200))
# input_a = tf.placeholder(shape=[None, 200], dtype=tf.int32)
# out = tf.expand_dims(input_a, axis=[-1])
# out = tf.reshape(out, [-1])
# sess = tf.Session()
# out = sess.run(out, feed_dict={input_a: a})
# print(out.shape)

a = tf.constant(np.array([[1, 1, 1, 1, 1, 0, 0]]), dtype=tf.float32)
b = tf.constant(np.ones(shape=[5, 1]), dtype=tf.float32)
c = a * b
sess = tf.Session()
out = sess.run(c)
print(out.shape)
print(out)