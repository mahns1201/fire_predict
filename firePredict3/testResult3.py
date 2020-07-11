import tensorflow as tf
import numpy as np

Data = np.loadtxt('total7Data.csv', delimiter=',', dtype=np.float32)

x_data = Data[700:, 0:-1]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(model)

    save_path = "./savedTotal7Data.cpkt"
    saver.restore(sess, save_path)

    # data = (x_data,)

    dict = sess.run(hypothesis, feed_dict={X: x_data})
    newList = np.ravel(dict)

    np.savetxt("resultTotal7DataResult.csv", newList, delimiter=',')