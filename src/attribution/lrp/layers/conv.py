# https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/python/modules/convolution.py
# https://github.com/VigneshSrinivasan10/interprettensor/blob/master/interprettensor/modules/convolution.py

from audioop import bias
from rsa import sign
import tensorflow as tf
from typing import Tuple
from .padding import same_pad, remove_padding


na = tf.newaxis # None
_small_number = 1e-32

def propagate_conv_layer(x, w, b, x_out, r, bias_factor, eps, strides, padding, level=None) -> Tuple[tf.Tensor, tf.Tensor]:

    #if 5 < level:
    #    gamma = 10
    #else:
    gamma = 0

    N,Hout,Wout,NF = r.shape
    hf,wf,df,NF = w.shape 
    hstride, wstride = strides # stride
    bias_nb_units = hf * wf * df

    Rsink = None
    if b is None:
        b = tf.zeros([NF], dtype=tf.float64)
    else:
        b = b[na,na,na,na,...]
        
        #sign_out = tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64)
        #denom = x_out + _small_number * sign_out
        #x_out = x_out - b

        #r1 = r * ((x_out + (x_out/denom) * _small_number * sign_out) / denom)
        #Rsink = r * ((b + (b/denom) * _small_number * sign_out) / denom)
        #tf.print("Bias change?:", tf.reduce_sum(r)-tf.reduce_sum(r1+Rsink))
        #tf.print("             ", tf.reduce_sum(r),tf.reduce_sum(r1),tf.reduce_sum(Rsink))
        #r = r1


    paddings = None
    pad_mask = tf.ones_like(x, dtype=tf.float64)
    if "same" == padding:
        x, paddings = same_pad(x, w, strides)
        
        pad_mask = tf.pad(pad_mask, paddings, "CONSTANT")
        pad_offset = (paddings[1][0], paddings[2][0])
    else:
        pad_offset = (0, 0)

    Rx = tf.zeros_like(x, dtype=tf.float64)
    w = w[na,...]
    
    x = tf.expand_dims(x, axis=-1)
    pad_mask = tf.expand_dims(pad_mask, axis=-1)

    sign_out = tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64)
    r_norm = r / (x_out + _small_number * sign_out + gamma * tf.math.maximum(x_out, 0.))
    #tf.print("paddings:", paddings)
    
    sign_out = tf.expand_dims(sign_out, -2)
    r_norm = tf.expand_dims(r_norm, -2)

    for i in range(Hout):
        i_cords_start = i * hstride
        i_cords_end = i_cords_start + hf
        for j in range(Wout):
            j_cords_start = j * wstride 
            j_cords_end = j_cords_start + wf

            temp_x = x[:,i_cords_start:i_cords_end,j_cords_start:j_cords_end,:,:]
            temp_pad_mask = pad_mask[:,i_cords_start:i_cords_end,j_cords_start:j_cords_end,:,:]
            #temp_r = r[:,i:i+1,j:j+1,:,:]
            #temp_x_out = x_out[:,i:i+1,j:j+1,:]
            temp_sign_out = sign_out[:,i:i+1,j:j+1,:]
            temp_r_norm = r_norm[:,i:i+1,j:j+1,:]
            
            Z = w * temp_x + w * gamma * tf.math.maximum(temp_x, 0.)
            Z_eps = bias_factor * (temp_pad_mask * (b + _small_number * temp_sign_out)) / tf.reduce_sum(temp_pad_mask, axis=(1,2,3,4),keepdims=True)
            Z = Z + Z_eps

            new_r = Z * temp_r_norm
            new_r = tf.reduce_sum(new_r, axis=-1)
        
            temp_paddings = tf.constant([
                [0, 0],
                [i_cords_start, Rx.shape[1] - i_cords_end],
                [j_cords_start, Rx.shape[2] - j_cords_end],
                [0, 0]
            ])
            new_r = tf.pad(new_r, temp_paddings, "CONSTANT")
            Rx = Rx + new_r

    #tf.print("r change?:", tf.reduce_sum(r)-tf.reduce_sum(Rx),tf.reduce_sum(r_norm))

    if paddings is not None:
        start_r = tf.reduce_sum(Rx)
        Rx = remove_padding(Rx, paddings)
        padSink = start_r - tf.reduce_sum(Rx)
        #tf.print("padSink",padSink)
        Rsink = padSink if Rsink is None else (Rsink + padSink)
        #TODO assert pad sink == 0

    return Rx, Rsink
