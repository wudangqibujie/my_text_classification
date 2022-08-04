import os
import pandas as pd
import numpy as np
import tensorflow as tf


FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


dir_path = os.path.dirname(os.path.realpath(__file__))
train_path=os.path.join(dir_path,'iris_training.csv')
test_path=os.path.join(dir_path,'iris_test.csv')


train = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train, train.pop('Species')
print(train_y.unique())
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()


def get_batch(batch_size=8):
    bt = train_x.shape[0]//batch_size
    for i in range(bt):
        batch_x, batch_y = train_x[batch_size*i: batch_size*(i+1)], train_y[batch_size*i: batch_size*(i+1)]
        yield batch_x, batch_y


input_X = tf.placeholder(dtype=tf.float64, shape=[None, 4])
input_Y = tf.placeholder(dtype=tf.int64, shape=[None])

dnn1 = tf.layers.dense(input_X, units=24, activation=tf.sigmoid)
dnn2 = tf.layers.dense(dnn1, units=3)
# W1 = tf.get_variable(name="W1", shape=[4, 8], dtype=tf.float32, initializer=tf.initializers.random_uniform)
# W1 = tf.Variable(tf.truncated_normal([4,8],stddev=0.1))
# b1 = tf.Variable(tf.constant(0., shape=[8]))
# b1 = tf.get_variable(name="b1", shape=[8], dtype=tf.float32, initializer=tf.initializers.zeros)
# W2 = tf.get_variable(name="W2", shape=[8, 3], dtype=tf.float32, initializer=tf.initializers.random_uniform)
# W2 = tf.Variable(tf.truncated_normal([8, 3],stddev=0.1))
# b2 = tf.Variable(tf.constant(0., shape=[3]))
# b2 = tf.get_variable(name="b2", shape=[3], dtype=tf.float32, initializer=tf.initializers.zeros)


# out = tf.matmul(input_X, W1)
# out = tf.add(out, b1)
# out = tf.nn.sigmoid(out)
# out = tf.matmul(out, W2)
# out = tf.add(out, b2)
out = tf.nn.softmax(dnn2)
loss = tf.losses.sparse_softmax_cross_entropy(labels=input_Y, logits=out)
init = tf.global_variables_initializer()

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1)
obj = train_op.minimize(loss)


metrics = tf.argmax(dnn2, axis=1)


if __name__ == '__main__':
    for epoch in range(100):
        with tf.Session() as sess:
            sess.run(init)
            for x, y in get_batch():
                sess.run(obj, feed_dict={input_X: x, input_Y: y})
            print(f"EPOCH {epoch}", sess.run(loss, feed_dict={input_X: x, input_Y: y}))
            print("预测", sess.run(metrics, feed_dict={input_X: x, input_Y: y}))
            print("真实", y)