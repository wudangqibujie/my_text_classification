import numpy as np
import tensorflow as tf



train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
print(train_X.shape, train_Y.shape)
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(1.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X * w - b)
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(0.01)
grads = train_op.compute_gradients(loss, [w, b])
for i, (grd, var_) in enumerate(grads):
    if grd is not None:
        grads[i] = (tf.clip_by_norm(grd, 3), var_)
train_op = train_op.apply_gradients(grads)
global_step = tf.train.get_or_create_global_step()
# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epoch = 1
    for i in range(20):
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X:x, Y: y})
        print(sess.run(loss, feed_dict={X: x, Y: y}))
        global_step = global_step + 1
        print(sess.run(global_step))
        #     _, ls, w_value, b_value = sess.run([train_op, loss, w, b], feed_dict={X: x, Y: y})
        # print("Epoch: {}, w: {}, b: {} loss:{}".format(epoch, w_value, b_value, ls))
        # epoch += 1
