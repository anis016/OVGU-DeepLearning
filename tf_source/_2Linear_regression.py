"""
Linear regression on the California housing data
Objective function, theta = (X^T . X)^-1 . X^T . y
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from generate_batch import BatchGen

n_epochs = 100
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape # 20640(m) examples with 8(n) features

scaler = StandardScaler() # we need to normalize the input feature vector when using gradient descent
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# print(scaled_housing_data_plus_bias[0]) # see how the 1st example looks like
# print(housing.target[0]) # label of the 1st optimizer

# https://stackoverflow.com/questions/10894323/what-does-the-c-underscore-expression-c-do-exactly#answer-51884244
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# print(housing.data[:2], "\n")
# print(housing.target[:2])
# print(housing_data_plus_bias[:2])
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as session:
#     theta_value = theta.eval()
#     print(theta_value)


## passing the whole dataset 1 time.
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta') # initialize theta with random
# y_pred = tf.matmul(X, theta, name='predictions')
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name='mse')
# # gradients = 2/m * tf.matmul(tf.transpose(X), error) # manually doing gradient calculations
# gradients = tf.gradients(mse, [theta])[0] # using tensorflow's autodiff
# # training_op = tf.assign(theta, theta - learning_rate * gradients) # manually doing theta calculations
# training_op = optimizer.minimize(mse) # using tensorflow's Gradient Descent optimizer
#
# init = tf.global_variables_initializer()
# with tf.Session() as session:
#     session.run(init)
#     for epoch in range(n_epochs + 1):
#         if epoch % 10 == 0:
#             print("Epoch", epoch, "MSE =", mse.eval())
#
#         session.run(training_op)
#
#     best_theta = theta.eval()
#     print(best_theta)

## mini-batch gradient descent
batch_size = 128
n_batches = int(np.ceil(m / batch_size))

X = tf.placeholder(dtype=tf.float32, shape=(None, n + 1), name='X') # total n features + 1 bias
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
mse_summary = tf.summary.scalar('MSE', mse)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]

    return X_batch, y_batch

init = tf.global_variables_initializer()
dataset = BatchGen(scaled_housing_data_plus_bias, housing.target)
with tf.Session() as session:
    session.run(init)

    for epoch in range(n_epochs):
        loss = 0.0
        for batch_index in range(n_batches):
            X_batch, y_batch = dataset.next_batch(batch_size, shuffle=True)
            y_batch = y_batch.reshape(-1, 1) # reshape the label to match the shape: (batch_size, 1)

            # X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            session.run(optimizer, feed_dict={X: X_batch, y: y_batch})
            loss += mse.eval(feed_dict={X: X_batch, y: y_batch})

        print("loss: ", loss/batch_size)

    best_theta = theta.eval()
    print(best_theta)