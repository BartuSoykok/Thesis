
import tensorflow as tf
from typing import Tuple
from .padding import same_pad, remove_padding

na = tf.newaxis # None
_small_number = 1e-32


def propagate_pooling_layer(x, pool_size, x_out, r, strides, padding) -> Tuple[tf.Tensor, tf.Tensor]:
    
    N,H,W,D = x.shape
    N,Hout,Wout,NF = r.shape
    hf,wf = pool_size
    hstride, wstride = strides # stride
    
    paddings = None
    pad_mask = tf.ones_like(x, dtype=tf.float64)
    if "same" == padding:
        x, paddings = same_pad(x, pool_size, strides)     # TODO Drop padded zeros afterwards
        pad_mask = tf.pad(pad_mask, paddings, "CONSTANT")
        pad_offset = (paddings[1][0], paddings[2][0])
    else:
        pad_offset = (0, 0)
        
    Rx = tf.zeros_like(x, dtype=tf.float64)
    
    #Hout = (H - hf) // hstride + 1
    #Wout = (W - wf) // wstride + 1
    for i in range(Hout):
        i_cords_start = i * hstride
        i_cords_end = i_cords_start + hf
        for j in range(Wout):
            j_cords_start = j * wstride
            j_cords_end = j_cords_start + wf
            
            temp_x = x[:,i_cords_start:i_cords_end,j_cords_start:j_cords_end,:]
            temp_pad_mask = pad_mask[:,i_cords_start:i_cords_end,j_cords_start:j_cords_end,:]
            temp_r = r[:,i:i+1,j:j+1,:]
            temp_x_out = x_out[:,i:i+1,j:j+1,:]

            numerator = tf.cast(tf.math.equal(temp_x,temp_x_out), dtype=tf.float64)
            denom = tf.reduce_sum(numerator, axis=(1,2), keepdims=True)

            sign_out = tf.cast(tf.where(denom >= 0, 1., -1.), tf.float64)
            numerator = numerator + _small_number * sign_out
            denom = denom + _small_number * sign_out

            num = numerator / denom
            new_r = num * temp_r
            #tf.print("new_r change:", tf.reduce_sum(new_r)-tf.reduce_sum(temp_r))
            
            temp_paddings = tf.constant([
                [0, 0],
                [i_cords_start, Rx.shape[1] - i_cords_end],
                [j_cords_start, Rx.shape[2] - j_cords_end],
                [0, 0]
            ])
            new_r = tf.pad(new_r, temp_paddings, "CONSTANT")
            Rx = Rx + new_r
    #tf.print("pool r change:", tf.reduce_sum(r)-tf.reduce_sum(Rx))
    
    #tf.print(Rx.shape)

    if paddings is not None:
        start_r = tf.reduce_sum(Rx)
        Rx = remove_padding(Rx, paddings)
        Rsink = start_r - tf.reduce_sum(Rx)
        tf.print(tf.reduce_sum(Rsink))
        return Rx, Rsink
    else:
        return Rx, None

def propagate_global_pooling_layer(x, x_out, r) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TODO this is only avg
    """

    #print(x.shape, "!=", x_out.shape)
    if len(x.shape) != len(x_out.shape):
        x_out = x_out[:, na, na, :]
        r = r[:, na, na, :]
        #print(r.shape, "--", x_out.shape)
    batch,h,w,c = x.shape

    sign_out = tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64)
    denom = x_out * (h * w) + _small_number * sign_out # Avg * Dim

    num = x / denom

    new_r = num * r

    return new_r, None

