
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow_addons as tfa
def peakiness_score(inputs, ksize=3, dilation=1,name='con'):

    pad_size = ksize // 2 + (dilation - 1)
    pad_inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size],[pad_size, pad_size], [0, 0]], mode='REFLECT')

    avg_spatial_inputs = tf.nn.pool(pad_inputs, [ksize, ksize],pooling_type='AVG', padding='VALID', dilations=[dilation, dilation])
    avg_channel_inputs = tf.reduce_mean(inputs, axis=-1, keepdims=True)

    alpha = tf.math.softplus(inputs - avg_spatial_inputs)
    beta = tf.math.softplus(inputs - avg_channel_inputs)
    return alpha, beta
