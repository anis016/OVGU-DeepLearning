import tensorflow as tf
import numpy as np

total_data = 10
batch_size = 5
epochs     = 2
n_batch    = int(total_data / batch_size)

def one_shot_iterator_example():

    X = np.arange(total_data)
    dataset = tf.data.Dataset.from_tensor_slices(X)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(total_data)

    # create an one_shot iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # make_one_shot_iterator will throw error after 1st epoch because it is exhausted
    # one way is to use dataset.repeat() or make_initializable_iterator
    with tf.Session() as sess:
        for epoch in range(epochs):
            for batch in range(n_batch):
                val = sess.run(next_element)
                print(val)
            print("{0} epoch done\n".format(epoch + 1))

def initializable_iterator_example():
    X = np.arange(total_data)
    dataset = tf.data.Dataset.from_tensor_slices(X)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    # create an initializable iterator
    iterator_init = dataset.make_initializable_iterator()
    next_element_init = iterator_init.get_next()

    with tf.Session() as sess:
        sess.run(iterator_init.initializer)

        for epoch in range(epochs):
            for batch in range(n_batch):
                val = sess.run(next_element_init)
                print(val)
            print("{0} epoch done\n".format(epoch + 1))

            # after first run re-initialize the iterator. this is not possible in make_one_shot_iterator.
            # however, if used dataset.repeat() then we can ignore this line.
            sess.run(iterator_init.initializer)

def simple_zip_example():
    x = np.arange(10)
    y = np.arange(1, 11)

    # create dataset objects from the arrays
    dataset_X = tf.data.Dataset.from_tensor_slices(x)
    dataset_y = tf.data.Dataset.from_tensor_slices(y)

    # zip the two datasets together
    dataset_comb = tf.data.Dataset.zip((dataset_X, dataset_y))
    # make a batch
    dataset_comb = dataset_comb.batch(batch_size)
    dataset_comb = dataset_comb.repeat()

    # create an iterator
    iterator = dataset_comb.make_initializable_iterator()
    # extract next element
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for epoch in range(epochs):
            for batch in range(n_batch):
                val = sess.run(next_element)
                print(val)
            print("{0} epoch done\n".format(epoch + 1))

if __name__ == '__main__':
    # one_shot_iterator_example()
    # initializable_iterator_example()
    simple_zip_example()