import numpy as np
import tensorflow as tf

# RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类BasicRNNCell和BasicLSTMCell。顾名思义，前者是RNN的基础类，后者是LSTM的基础类。

# rnn_cell1 = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# rnn_cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# rnn_cell3 = tf.nn.rnn_cell.LSTMCell(num_units=128)
# rnn_cell4 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=256) for _ in range(3)])
#
# cell = rnn_cell4
# x = tf.placeholder(dtype=tf.float32, shape=[None, 100, 128])
# dyn_batch_size = tf.shape(x)[0]
# h0 = cell.zero_state(dyn_batch_size, dtype=tf.float32)
# # out, h1 = cell.__call__(x, h0)
# out = tf.nn.dynamic_rnn(cell=cell, inputs=x, initial_state=h0, dtype=tf.float32)


class TextRNNLayer:
    def __init__(self, input_x, config, module_name=None):
        self.config = config
        self.module_name = module_name
        self.input_x = input_x
        self.build()

    def build(self):
        rnn_num_layers = self.config["num_layer"]
        if self.config["bidirectional"]:
            X = self.input_x
            for n in range(rnn_num_layers):
                fw_cell = self._get_cell()
                bw_cell = self._get_cell()
                (out_fw, out_bw), (h_fw, h_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=X, dtype=tf.float32)
                X = tf.concat([out_fw, out_bw], axis=2)
            return X
        if rnn_num_layers == 1:
            cell = self._get_cell()
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([self._get_cell() for _ in range(rnn_num_layers)])
        dyn_batch_size = tf.shape(self.input_x)[0]
        h0 = cell.zero_state(dyn_batch_size, dtype=tf.float32)
        self.raw_out = tf.nn.dynamic_rnn(cell=cell, inputs=self.input_x, initial_state=h0, dtype=tf.float32)

    def _get_cell(self):
        cell_type = self.config["cell_type"]
        num_units = self.config["num_unite"]
        if cell_type == "rnn":
            return tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(num_units=num_units)
        else:
            raise "RNN Type not found!"

    def get_output(self):
        if self.config["cell_type"] == "rnn":
            pass

    def get_hidden_output(self):
        pass


# if __name__ == '__main__':
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     rslt_out = sess.run(out, feed_dict={x: np.random.random((64, 100, 128))})
#     print(rslt_out)
#     print(len(rslt_out), rslt_out[0].shape)
#     print([i.c.shape for i in rslt_out[1]])
#     print([i.h.shape for i in rslt_out[1]])




