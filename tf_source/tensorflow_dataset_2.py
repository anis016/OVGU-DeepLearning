import tensorflow as tf
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(42)

## for tensorboard visualization
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{0}/run-{1}".format(root_logdir, now)
file_writer = tf.summary.FileWriter(logdir)
##

learning_rate  = 0.01
batch_size     = 128
training_epoch = 100

def get_mnist_data():

    dirname, _ = os.path.split(os.path.dirname(os.path.abspath("__file__")))
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

def nn_model(inputs, is_training):
    with tf.name_scope("Dense_NET"):
        fc1 = tf.layers.dense(inputs, units=256, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)
        fc2 = tf.layers.dense(fc1, units=256, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc2, rate=0.4, training=is_training)
        out = tf.layers.dense(fc2, units=10)

        return out

# get the normalized MNIST data
X_train, y_train, X_test, y_test = get_mnist_data()

# convert sequence label to one-hot encoded label
onehot_encoder = OneHotEncoder(categorical_features = [0]) # [0] -> which axis to be one-hot encoded?
y_train = onehot_encoder.fit_transform(y_train).toarray()
y_test  = onehot_encoder.fit_transform(y_test).toarray()

with tf.variable_scope("datasets"):
    # placeholder graph for the input
    training_batch_size  = tf.placeholder(tf.int64)
    inference_batch_size = tf.placeholder(tf.int64)
    handle = tf.placeholder(dtype=tf.string, shape=[]) # feeding this into feed_dict
    is_training = tf.placeholder(dtype=tf.bool)  # feeding this into feed_dict

    # create the training dataset
    dataset_X_train = tf.data.Dataset.from_tensor_slices(X_train)
    dataset_y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((dataset_X_train, dataset_y_train))\
        .shuffle(50000, reshuffle_each_iteration=True)\
        .repeat()\
        .batch(training_batch_size)

    dataset_X_valid = tf.data.Dataset.from_tensor_slices(X_test)
    dataset_y_valid = tf.data.Dataset.from_tensor_slices(y_test)
    valid_dataset = tf.data.Dataset.zip((dataset_X_valid, dataset_y_valid))\
        .repeat()\
        .batch(inference_batch_size)

    # generalized iterator
    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)
    train_iterator = train_dataset.make_initializable_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()

X_instance, y_label = iterator.get_next()
logits = nn_model(X_instance, is_training)

# add the optimizer and the loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=logits))
    tf.summary.scalar("Loss", loss)

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# get accuracy
with tf.variable_scope("inference"):
    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# merge all the summary
merged_summary = tf.summary.merge_all()

# initialize the variables
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.graph.finalize()
    session.run(init)

    # Initialize the iterator
    session.run([train_iterator.initializer, valid_iterator.initializer],
                feed_dict={training_batch_size: batch_size,
                           inference_batch_size: 500})
    train_handle, valid_handle = session.run([train_iterator.string_handle(),
                                              valid_iterator.string_handle()])

    # Run the initializer
    for epoch in range(1, training_epoch + 1):
        session.run([optimizer], feed_dict={handle: train_handle,
                                            is_training: True})

        if epoch % 10 == 0:
            # Calculate the loss and accuracy
            loss_local, acc, summary = session.run([loss, accuracy, merged_summary],
                                                   feed_dict={handle: train_handle,
                                                              is_training: True})

            file_writer.add_summary(summary, global_step=epoch)
            file_writer.flush()

            valid_acc, valid_loss = [], []
            for _ in range(20):
                tmp = session.run([accuracy, loss], feed_dict={handle: valid_handle,
                                                              is_training: False})
                valid_acc.append(tmp[0])
                valid_loss.append(tmp[1])
            valid_loss = np.mean(valid_loss)
            valid_acc  = np.mean(valid_acc)

            print("Epoch: {0:3d}, Train Loss: {1:.4f}, Train Accuracy: {2:.4f}% "
                  "Valid Loss: {3:.4f}, Valid Accuracy: {4:.4f}%"
                  .format(epoch, loss_local, acc * 100, valid_loss, valid_acc * 100))

file_writer.close()