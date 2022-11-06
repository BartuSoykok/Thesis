import tensorflow as tf


_small_number = 1e-32

def propagate_dense_layer(x, w, b, x_out, r, bias_factor, eps) -> tf.Tensor:
    #tf.print("dense start", tf.reduce_sum(r))
    bias_nb_units = w.shape[0]
    w = tf.expand_dims(w, axis=0)
    x = tf.expand_dims(x, axis=-1)
    x_out = tf.expand_dims(x_out, -2)
    r = tf.expand_dims(r, -2)
    if b is None:
        b = tf.zeros([w.shape[-1]], dtype=tf.float64)

    b = b[tf.newaxis,tf.newaxis,...]

    #x_out_without_b = x_out - (bias_factor * b)
    sign_out = tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64)
    Z_b = bias_factor * (b + eps * sign_out) / bias_nb_units
    #Z_b = tf.expand_dims(Z_b, 1)

    Z = w * x
    Z = Z + Z_b
    Zs = x_out + eps * sign_out
    #Zs = x_out_without_b + eps * tf.cast(tf.where(x_out_without_b >= 0, 1., -1.), tf.float64)

    new_r = (Z / Zs) * r

    r_prev = tf.reduce_sum(new_r, axis=-1)
    
    #tf.print("dense end", tf.reduce_sum(r_prev))
    return r_prev
