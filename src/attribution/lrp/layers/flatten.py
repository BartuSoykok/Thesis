import tensorflow as tf
from typing import Tuple

def propagate_flatten_layer(x, r) -> tf.Tensor:
    r_prev = tf.reshape(r, x.shape)
    return r_prev
