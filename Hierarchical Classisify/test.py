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

# x = tf.constant(np.ones(shape=[64, 12, 100, 10]), dtype=tf.float32)
# out = tf.nn.softmax(x)

# print(out.shape)
# print(out)

# debug = tf.range(0, 64, dtype=tf.int32) * 200
# sess = tf.Session()
# out = sess.run(debug)
# print(out)

init_checkpoint = r"E:\NLP——project\uncased_L-12_H-768_A-12\bert_model.ckpt"
init_vars = tf.train.list_variables(init_checkpoint)
print(init_vars)


