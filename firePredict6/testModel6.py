import tensorflow as tf
import numpy as np

data = np.loadtxt('wonju234.csv', delimiter=',', dtype=np.float32)

x_data = data[:, 0:-1]
y_data = data[:, [-1]]

# Make sure the shape and data are OK
# print(x_data, "\nx_data shape:", x_data.shape)
# print(y_data, "\ny_data shape:", y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10000 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

saver = tf.train.Saver()
save_path = saver.save(sess, "./savedwonju234.cpkt")
print("학습된 모델을 저장했습니다.")