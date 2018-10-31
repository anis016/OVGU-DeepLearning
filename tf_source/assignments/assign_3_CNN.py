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
root_logdir = "logs"
logdir = "{0}/run-{1}".format(root_logdir, now)
file_writer = tf.summary.FileWriter(logdir)
##

learning_rate  = 0.01
batch_size     = 128
training_epoch = 100

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

def cnn_model(inputs, is_training):
    with tf.name_scope("Conv_NET"):

        ## Input layer
        # MNIST data input is a 1-D vector of 784 features.
        # Reshape each example to match the format: [batch_size X img_height X img_width X channel]

        with tf.name_scope("inputs"):
            input_layer = tf.reshape(inputs, shape=[-1, 28, 28, 1])

            # this will display the weights of the first 10 hidden units
            tf.summary.image("weights", input_layer, max_outputs=10)

            zeros = tf.zeros_like(input_layer)
            weights_pos = tf.nn.relu(input_layer)
            weights_neg = tf.nn.relu(-input_layer)

            # we concatenate along the channel axis, using zeros for the third channel
            weights_3channel = tf.concat((weights_pos, weights_neg, zeros), axis=3)
            tf.summary.image("weights_hack", weights_3channel, max_outputs=10)

        ## Convolutional Layer and Pooling Layer#1
        # Convolution layer with 32 filters with kernel size [5 X 5] with ReLU activation function.
        with tf.variable_scope("conv1"):
            conv_layer1 = tf.layers.Conv2D(
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            conv1 = conv_layer1.apply(inputs=input_layer)
            # Max pooling with filter size [2 X 2] and stride of 2 (specifies pooled region do not overlap)
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[2, 2],
                strides=2
            )

        ## Convolutional Layer and Pooling Layer#2
        # Convolution layer with 64 filters with kernel size [5 X 5] with ReLU activation function.
        with tf.name_scope("conv2"):
            conv_layer2 = tf.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            conv2 = conv_layer2.apply(inputs=pool1)
            # Max pooling with filter size [2 X 2] and stride of 2 (specifies pooled region do not overlap)
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=[2, 2],
                strides=2
            )

        # Flatten the data to a 1-D vector for the Dense layer
        with tf.name_scope("fully_connected"):
            fc = tf.layers.flatten(pool2)

        # Dense Layer
        dense = tf.layers.dense(inputs=fc, units=1024, activation=tf.nn.relu)
        dense = tf.layers.dropout(inputs=dense, rate=0.4, training=is_training)

        # Output Layer for MNIST 10 class prediction
        out = tf.layers.dense(inputs=dense, units=10)

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
logits = cnn_model(X_instance, is_training)

# add the optimizer and the loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=logits))
    tf.summary.scalar("cross_entropy", loss)  # add scalar summary

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# get accuracy
with tf.variable_scope("inference"):
    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)  # add scalar summary

# merge all the summary
merged_summary = tf.summary.merge_all()

# initialize the variables
init = tf.global_variables_initializer()
with tf.Session() as session:
    file_writer.add_graph(session.graph)  # adding the summary graph
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