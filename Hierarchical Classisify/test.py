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
a = np.ones(shape=(10, 20, 30))
input_a = tf.placeholder(shape=[None, 20, 30], dtype=tf.int32)
sh = tf.shape(input_a)
sess = tf.Session()
out, out_sh = sess.run([input_a, sh], feed_dict={input_a: a})
print(a.shape, type(a.shape))
print(out.shape, type(out))
print(out_sh)
