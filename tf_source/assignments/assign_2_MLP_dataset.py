import tensorflow as tf
import os
import glob
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(42)

learning_rate  = 0.01
batch_size     = 128
training_epoch = 50

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

def nn_model(inputs):
    fc1 = tf.layers.dense(inputs, units=256, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, units=256, activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc2, units=10)

    return fc3

# get the normalized MNIST data
X_train, y_train, X_test, y_test = get_mnist_data()

# convert sequence label to one-hot encoded label
onehot_encoder = OneHotEncoder(categorical_features = [0]) # [0] -> which axis to be one-hot encoded?
y_train = onehot_encoder.fit_transform(y_train).toarray()
y_test  = onehot_encoder.fit_transform(y_test).toarray()

# create the training dataset
dataset_X_train = tf.data.Dataset.from_tensor_slices(X_train)
dataset_y_train = tf.data.Dataset.from_tensor_slices(y_train)
train_dataset = tf.data.Dataset.zip((dataset_X_train, dataset_y_train))\
    .shuffle(500)\
    .repeat()\
    .batch(batch_size)

dataset_X_valid = tf.data.Dataset.from_tensor_slices(X_test)
dataset_y_valid = tf.data.Dataset.from_tensor_slices(y_test)
valid_dataset = tf.data.Dataset.zip((dataset_X_valid, dataset_y_valid))\
    .shuffle(500)\
    .repeat()\
    .batch(batch_size)

# create one shot iterator
iterator_train = train_dataset.make_one_shot_iterator()
training_init_op = iterator_train.get_next()

iterator_valid = valid_dataset.make_one_shot_iterator()
validation_init_op = iterator_valid.get_next()

logits = nn_model(training_init_op[0])

# add the optimizer and the loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=training_init_op[1], logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# get accuracy
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(training_init_op[1], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.graph.finalize()
    session.run(init)

    # training cycle
    # initialize the iterator
    session.run(training_init_op)
    for epoch in range(training_epoch + 1):
        l, _, acc = session.run([loss, optimizer, accuracy])
        if epoch % 10 == 0:
            print("Epoch: {0:2d}, loss: {1:.4f}, training accuracy: {2:.4f}%".format(epoch, l, acc * 100))

    # validation cycle
    # re-initialize the iterator
    session.run(validation_init_op)

    avg_acc = 0
    for epoch in range(10):
        acc = session.run([accuracy])
        avg_acc += acc[0]

    acc = avg_acc / 10
    print("\nAverage validation accuracy: {0:.4f}%".format(acc * 100))