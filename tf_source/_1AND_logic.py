"""
Task: implement AND logic
"""

import tensorflow as tf

T, F = 1.0, -1.0 # True has +1.0 value and False has -1.0 value
training_input = [
    [T, T],
    [T, F],
    [F, T],
    [F, F]
]

training_output = [
    [T],
    [F],
    [F],
    [F]
]

W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([4]), dtype=tf.float32)

