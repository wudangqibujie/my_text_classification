# import random
# import re
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
# sen = "弱小的我也有大梦想"
# tokens = tokenizer.tokenize_jay(sen)
# print(tokens)
# import tensorflow as tf
# import numpy as np
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

# init_checkpoint = r"E:\NLP——project\uncased_L-12_H-768_A-12\bert_model.ckpt"
# init_vars = tf.train.list_variables(init_checkpoint)
# print(init_vars)


from tokenize_jay.pure_tokenize import Corpus, BaseTokenizer
from dataset.dataset import DataSet, TextClfTFRecord
import os
base_dir = "../../THUCNews"
# dirs = [i for i in os.listdir(base_dir) if "." not in i]
cate = "体育"

files = os.listdir(os.path.join(base_dir, cate))
needed_files = [os.path.join(base_dir, cate, i) for i in files[: 10000]]


def parse(line):
    return line.strip()

label_info = dict()
flg = 0
for c in [i for i in os.listdir(base_dir) if "." not in i]:
    label_info[c] = flg
    flg += 1

# corpus = Corpus(needed_files, "./util/pure_vocab.txt", parse_line=parse)
# corpus.build_vocab()
# corpus.write_vocab()
# from tqdm import tqdm
# print(label_info)
# tokenize = BaseTokenizer('./util/pure_vocab.txt', './util/stopword.dic')
out_tf_file = [f'../../data/THU/{i}.tfrecord' for i in range(100)]
# dataSet = DataSet(tokenize, 100, label_info, out_tf_file)
# for file in tqdm(needed_files):
#     with open(os.path.join(base_dir, cate, file), encoding="utf-8") as f:
#         for i in f:
#             text = i.strip()
#             if not text:
#                 continue
#             tokens = tokenize.tokenize(text)
#             # print(text)
#             # print(tokens)
#             ids = tokenize.convert_tokens_to_ids(tokens)
#             reverse_tokens = tokenize.convert_ids_to_tokens(ids)
#             # print(ids)
#             # print(reverse_tokens)
#             dataSet.write_tfrecord(ids, "体育")

textClfTFRecord = TextClfTFRecord(out_tf_file, 64, 100)
textClfTFRecord.get_batch()
import tensorflow as tf
sess = tf.Session()

for dataset in textClfTFRecord.datasets:
    while True:
        try:
            batch_data = sess.run(dataset)
            input_ids = batch_data["input_ids"]
            label_code = batch_data["label_code"]
            print(input_ids.shape, label_code.shape)
        except tf.errors.OutOfRangeError:
            print("READ OVER")
            break
