import tensorflow as tf
import numpy as np
from data import Tokenize, Multi_H_Dataset
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = Tokenize("vocab.txt")
dataset = Multi_H_Dataset(["Sample_data/sample_train.txt"], 3, tokenizer)


class FastText:
    def __init__(self):
        pass


class TextCNNConfig:
    embedding_dim = 64
    seq_length = 120
    first_cls_num, sec_cls_num, trd_cls_num = dataset.get_label_num
    num_filters = 256
    kernal_size = 5
    vocab_size = tokenizer.vocab_num
    hidden_size = 125,
    dropout_keep_rt = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epoch = 10


class TextCNN:
    def __init__(self, config):
        self.config = config

        self.input_X = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name="input")
        self.input_fst_y = tf.placeholder(tf.float32, shape=[None, self.config.first_cls_num], name="first_cls")
        self.input_sec_y = tf.placeholder(tf.float32, shape=[None, self.config.sec_cls_num], name="second_cls")
        self.input_trd_y = tf.placeholder(tf.float32, shape=[None, self.config.sec_cls_num], name="tfird_cls")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

    def initial_model(self):
        embedding = tf.get_variable(name="embedding", shape=[self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_X)
        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters ,self.config.kernel_size, name="conv1")
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")

        with tf.name_scope("score"):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.contrib.leyers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.first_cls_num, name="fc2")
            self.first_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optim"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_fst_y, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("acc"):
            correct_pred = tf.equal(tf.argmax(self.input_fst_y, 1), self.first_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    # config = TextCNNConfig()
    # print(config.first_cls_num)
    # model = TextCNN(config)
    for bt_x, bt_y, bt_text, bt_l_t in dataset.get_batch():
        print(bt_x)
        print(bt_y)
        print(np.array(bt_x).shape)
        # print(bt_text)
        # print(bt_l_t)
        # print(len(bt_l_t))
    print(dataset.get_label_num)
    # print(tokenizer.vocab_num)
    # print(dataset.length_info)