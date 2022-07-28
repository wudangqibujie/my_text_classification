import tensorflow as tf
import numpy as np
from data import Tokenize, Multi_H_Dataset
import matplotlib.pyplot as plt
import tensorflow.contrib.rnn as rnn
import seaborn as sns

tokenizer = Tokenize("vocab.txt")
dataset = Multi_H_Dataset(["Sample_data/sample_train.txt"], 3, tokenizer)


class FastText:
    def __init__(self):
        pass


class TextCNNConfig:
    embedding_dim = 64
    seq_length = 120
    first_cls_num, sec_cls_num, trd_cls_num = [36, 251, 831]
    num_filters = 256
    kernel_size = 5
    vocab_size = tokenizer.vocab_num
    hidden_dim = 125
    dropout_keep_rt = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epoch = 50


class TextCNN:
    def __init__(self, config):
        self.config = config
        self.input_X = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name="input")
        self.input_fst_y = tf.placeholder(tf.int32, shape=[None, ], name="first_cls")
        self.input_sec_y = tf.placeholder(tf.int32, shape=[None, ], name="second_cls")
        self.input_trd_y = tf.placeholder(tf.int32, shape=[None, ], name="tfird_cls")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
        self.initial_model()

    def initial_model(self):
        embedding = tf.get_variable(name="embedding", shape=[self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_X)
        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name="conv1")
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")
        with tf.name_scope("score"):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)
            self.logits1 = tf.layers.dense(fc, self.config.first_cls_num, name="fc21")
            self.logits2 = tf.layers.dense(fc, self.config.sec_cls_num, name="fc22")
            self.logits3 = tf.layers.dense(fc, self.config.trd_cls_num, name="fc23")
            self.first_pred = tf.argmax(tf.nn.softmax(self.logits1), 1, output_type=tf.int32)
            self.sec_pred = tf.argmax(tf.nn.softmax(self.logits2), 1, output_type=tf.int32)
            self.trd_pred = tf.argmax(tf.nn.softmax(self.logits3), 1, output_type=tf.int32)
        with tf.name_scope("optim"):
            self.cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_fst_y, logits=self.logits1)
            self.cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_sec_y, logits=self.logits2)
            self.cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_trd_y, logits=self.logits3)
            self.debug1 = tf.reduce_mean(self.cross_entropy1)
            self.debug2 = tf.reduce_mean(self.cross_entropy2)
            self.debug3 = tf.reduce_mean(self.cross_entropy3)
            self.loss = tf.reduce_mean(self.cross_entropy1+self.cross_entropy2+self.cross_entropy3)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("acc"):
            correct_pred1 = tf.equal(self.input_fst_y, self.first_pred)
            correct_pred2 = tf.equal(self.input_sec_y, self.sec_pred)
            correct_pred3 = tf.equal(self.input_trd_y, self.trd_pred)
            self.acc1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
            self.acc2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
            self.acc3 = tf.reduce_mean(tf.cast(correct_pred3, tf.float32))


class TextRNNConfig:
    embedding_dim = 64
    seq_length = 120
    first_cls_num, sec_cls_num, trd_cls_num = [36, 251, 831]
    num_filters = 256
    kernel_size = 5
    vocab_size = tokenizer.vocab_num
    hidden_dim = 125
    dropout_keep_rt = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epoch = 50
    decay_rate = 0.9
    decay_steps = 1000
    is_training = True


class TextRNN:
    def __init__(self, config):
        self.config = config
        self.input_X = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name="input")
        self.input_fst_y = tf.placeholder(tf.int32, shape=[None, ], name="first_cls")
        self.input_sec_y = tf.placeholder(tf.int32, shape=[None, ], name="second_cls")
        self.input_trd_y = tf.placeholder(tf.int32, shape=[None, ], name="tfird_cls")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
        self.l2_lambda = 0.0001
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = config.decay_steps, config.decay_rate
        self.initial_model()

    def initial_model(self):
        embedding = tf.get_variable(name="embedding", shape=[self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_X)

        lstm_fw_cell = rnn.BasicLSTMCell(self.config.hidden_dim)
        lstm_bw_cell = rnn.BasicLSTMCell(self.config.hidden_dim)
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.config.dropout_keep_rt)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.config.dropout_keep_rt)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedding_inputs, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)
        self.output_rnn_last = output_rnn[:, -1, :]

        self.logits1 = tf.layers.dense(self.output_rnn_last, self.config.first_cls_num, name="fc21")
        self.logits2 = tf.layers.dense(self.output_rnn_last, self.config.sec_cls_num, name="fc22")
        self.logits3 = tf.layers.dense(self.output_rnn_last, self.config.trd_cls_num, name="fc23")

        self.first_pred = tf.argmax(tf.nn.softmax(self.logits1), 1, output_type=tf.int32)
        self.sec_pred = tf.argmax(tf.nn.softmax(self.logits2), 1, output_type=tf.int32)
        self.trd_pred = tf.argmax(tf.nn.softmax(self.logits3), 1, output_type=tf.int32)
        with tf.name_scope("loss"):
            self.cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_fst_y,
                                                                                 logits=self.logits1)
            self.cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_sec_y,
                                                                                 logits=self.logits2)
            self.cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_trd_y,
                                                                                 logits=self.logits3)
            self.debug1 = tf.reduce_mean(self.cross_entropy1)
            self.debug2 = tf.reduce_mean(self.cross_entropy2)
            self.debug3 = tf.reduce_mean(self.cross_entropy3)
            loss = tf.reduce_mean(self.cross_entropy1 + self.cross_entropy2 + self.cross_entropy3)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            self.loss = loss + l2_losses
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("acc"):
            correct_pred1 = tf.equal(self.input_fst_y, self.first_pred)
            correct_pred2 = tf.equal(self.input_sec_y, self.sec_pred)
            correct_pred3 = tf.equal(self.input_trd_y, self.trd_pred)
            self.acc1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
            self.acc2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
            self.acc3 = tf.reduce_mean(tf.cast(correct_pred3, tf.float32))


if __name__ == '__main__':
    config = TextCNNConfig()
    model = TextCNN(config)
    # config = TextRNNConfig()
    # model = TextRNN(config)

    sess = tf.Session()
    sess.run([tf.global_variables_initializer()])
    for epoch in range(config.num_epoch):
        cnt, correct_cnt1, correct_cnt2, correct_cnt3 = 0, 0, 0, 0
        tnt_loss = 0.
        for bt_x, bt_y, bt_text, bt_l_t in dataset.get_batch():
            bt_x = np.array(bt_x)
            bt_y = np.array(bt_y)
            bt_1_y = bt_y[:, 0]
            bt_2_y = bt_y[:, 1]
            bt_3_y = bt_y[:, 2]
            feed_d = {model.input_X: bt_x,
                      model.input_fst_y: bt_1_y,
                      model.input_sec_y: bt_2_y,
                      model.input_trd_y: bt_3_y,
                      model.dropout_keep_prob: config.dropout_keep_rt}
            l1, l2, l3, loss, acc1, acc2, acc3, _ = sess.run([model.debug1, model.debug2, model.debug3, model.loss, model.acc1, model.acc2, model.acc3, model.optim], feed_dict=feed_d)
            bt_num = bt_x.shape[0]
            cnt += bt_num
            tnt_loss += loss * bt_num
            correct_cnt1 += acc1 * bt_num
            correct_cnt2 += acc2 * bt_num
            correct_cnt3 += acc3 * bt_num
        print(epoch, tnt_loss / cnt, correct_cnt1 / cnt, correct_cnt2 / cnt, correct_cnt3 / cnt)


