import numpy as np
import tensorflow as tf
from math import ceil


def same_pad(x, kernel, strides):
    _, in_height, in_width, _ = x.shape
    stride_height, stride_width = strides
    if hasattr(kernel, "shape"):
        filter_height, filter_width, _, _ = kernel.shape
    elif isinstance(kernel, tuple):
        filter_height, filter_width = kernel
    else:
        raise AttributeError(f"kernel is ony allowed as tuple of ints or has shape; but found {type(kernel)}.")
    #print("filter_height, filter_width",filter_height, filter_width)
    #print("stride_height, stride_width",stride_height, stride_width)

    if (in_height % stride_height == 0):
        pad_along_height = max(filter_height - stride_height, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_height), 0)
        
    if (in_width % stride_width == 0):
        pad_along_width = max(filter_width - stride_width, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    #print(pad_along_height, pad_along_width)
    #print(pad_top,pad_bottom,pad_left,pad_right)

    paddings = tf.constant([
        [0, 0], 
        [pad_left, pad_right],  # distribute padding half left and half right
        [pad_top, pad_bottom],  # distribute padding half left and half right
        [0, 0]])
    result = tf.pad(x, paddings, "CONSTANT")
    #tf.print("PAD",x.shape, result.shape, paddings)
    
    return result, paddings


def remove_padding(x, padding):
    if isinstance(padding, tf.Tensor):
        pad_top = padding[2][0]
        pad_bottom = padding[2][1]
        pad_left = padding[1][0]
        pad_right = padding[1][1]
    elif isinstance(padding, int):
        pad_top, pad_bottom, pad_left, pad_right = padding, padding, padding, padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        horizontal, vertical = padding
        if isinstance(vertical, int) and isinstance(horizontal, int):
            pad_top, pad_bottom, pad_left, pad_right = vertical, vertical, horizontal, horizontal
        elif isinstance(vertical, tuple) and len(vertical) == 2 and isinstance(horizontal, tuple) and len(horizontal) == 2:
            (pad_top, pad_bottom), (pad_left, pad_right) = vertical, horizontal
        else:
            raise AttributeError(f"Padding type is ony allowed tf.Tensor, int, tuple of ints or tuple of tuple of ints; but found {type(padding)}.")
    else:
        raise AttributeError(f"Padding type is ony allowed tf.Tensor, int, tuple of ints or tuple of tuple of ints; but found {type(padding)}.")
    
    _, in_width, in_height, _ = x.shape
    x_cropped = x[:, pad_left:in_width - pad_right, pad_top:in_height - pad_bottom, :]
    
    #tf.print("REMOVE PAD",x.shape, x_cropped.shape, padding)
    return x_cropped

"""inp = tf.ones((1, 13, 13, 1))
filter = tf.ones((6, 6, 1, 1))
strides = [5, 5]
output3, padding = same_pad(inp, filter, strides)

output = tf.nn.conv2d(inp, filter, strides, padding="SAME")
tuple(output.shape)

# Equivalently, tf.pad can be used, since convolutions pad with zeros.
inp = tf.pad(inp, padding)
# 'VALID' means to use no padding in conv2d (we already padded inp)
output2 = tf.nn.conv2d(inp, filter, strides, padding='VALID')
print()
print(padding)
tf.print(output.shape,output2.shape,output3.shape)"""