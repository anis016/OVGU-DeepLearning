import tensorflow as tf
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(42)

##---
# includes top level module
dname, fname = os.path.split(os.path.dirname(os.path.abspath("__file__")))
sys.path.append(dname)
from generate_batch import BatchGen
##---

def get_mnist_data():

    dirname, _ = os.path.split(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
    data_dir = os.path.join(dirname, "data")
    MNIST_dir = os.path.join(data_dir, "MNIST")

    if os.path.exists(MNIST_dir):
        all_files = glob.glob(MNIST_dir + "/*.csv")
        train_path = all_files[0] if 'train' in all_files[0] else all_files[1]
        test_path  = all_files[0] if 'test' in all_files[0] else all_files[1]

        train = pd.read_csv(train_path, header=None)
        X_train = train.iloc[:,1:].values/255.0
        y_train = train.iloc[:,0].values

        test = pd.read_csv(test_path, header=None)
        X_test = test.iloc[:, 1:].values/255.0
        y_test = test.iloc[:, 0].values

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return X_train, y_train, X_test, y_test
    else:
        raise IOError("Path: {0} not found!".format(MNIST_dir))

IMG_SIZE    = 28
IMG_PIXELS = IMG_SIZE * IMG_SIZE

# network parameters
n_inputs   = IMG_PIXELS # MNIST data input
n_hidden_1 = 256 # number of features/neurons in the first layer
n_hidden_2 = 256 # number of features/neurons in the second layer
num_classes = 10 # number of outputs

# get the normalized MNIST data
X_train, y_train, X_test, y_test = get_mnist_data()

# convert sequence label to one-hot encoded label
onehot_encoder = OneHotEncoder(categorical_features = [0]) # [0] -> which axis to be one-hot encoded?
y_train = onehot_encoder.fit_transform(y_train).toarray()
y_test  = onehot_encoder.fit_transform(y_test).toarray()

# placeholder graph for the input
X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name='y')

def multilayer_perceptron(x, weights, biases):
    # 1st hidden layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # 2nd hidden layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# store weights & biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}

learning_rate  = 0.01
training_epoch = 10
batch_size     = 128
n_batches      = int(np.ceil(len(y_train) / batch_size))

# construct the model
y_pred = multilayer_perceptron(X, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test model
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

dataset = BatchGen(X_train, y_train)
with tf.Session() as session:
    session.graph.finalize()
    session.run(init)

    # training cycle
    for epoch in range(training_epoch):
        avg_cost = 0
        for batch_index in range(n_batches):
            X_batch, y_batch = dataset.next_batch(batch_size, shuffle=True)
            _, c = session.run([optimizer, cost], feed_dict={X: X_batch,
                                                             y: y_batch})

            # compute average loss
            avg_cost += c / n_batches

        # display logs per epoch step
        print('Epoch: {0}, cost: {1:.4f}'.format(epoch, avg_cost))
        print("Test Accuracy: {0:.4f}%\n".format(accuracy.eval(feed_dict={X: X_test,
                                                                          y: y_test}) * 100))

        # save_path = saver.save(session, 'models/mnist/mlp_mnist.ckpt')