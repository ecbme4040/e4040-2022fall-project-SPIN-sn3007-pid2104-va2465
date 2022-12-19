import numpy as np
import tensorflow as tf

#this partitions an input image into windows of win_size
def win_partition(x, win_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // win_size, win_size, W // win_size, win_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    wins = tf.reshape(x, shape=[-1, win_size, win_size, C])
    return wins

#Reverses the window
def win_reverse(wins, win_size, H, W, C):
    x = tf.reshape(wins, shape=[-1, H // win_size, W // win_size, win_size, win_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x

#helper function that implements drop path based on probability "drop_prob"
def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)

    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output