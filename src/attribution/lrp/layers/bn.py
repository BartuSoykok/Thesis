# Code based on implementation of Christoph Wehner

import tensorflow as tf
from typing import Tuple

from .utils import Rule

_stabilizer = tf.constant(1e-30, dtype=tf.float64) # so no true division by 0 happens

def propagate_bn_layer(x, x_out, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon, mode = Rule.HEURISTIC_RULE) -> Tuple[tf.Tensor, tf.Tensor]: 
    
    #return foo1(x, x_out, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon)
    
    if mode == Rule.HEURISTIC_RULE:
        return heuristic_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon)
    elif mode == Rule.OMEGA_RULE:
        return omega_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon)
    elif mode == Rule.IDENTITY_RULE:
        return identity_rule(R_out)
    elif mode == Rule.EPSILON_RULE:
        return epsilon_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon)
    elif mode == Rule.Z_RULE:
        return z_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon)
    else:
        raise AttributeError(f"BatchNormalization mode not implemented: {mode}")
    
    
def foo2(x, x_out, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon) -> Tuple[tf.Tensor, tf.Tensor]:
    

    num = x * (x_out - beta) * R_out
    denom = (x - moving_mu) * x_out
    denom = denom + epsilon * tf.cast(tf.where(denom >= 0, 1., -1.), tf.float64)

    R_in = num / denom

    return R_in, None

def foo1(x, x_out, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon) -> Tuple[tf.Tensor, tf.Tensor]:

    a1 = x - (x / tf.reduce_sum(x, axis=-1, keepdims=True))
    a2 = (gamma / tf.sqrt(moving_var + epsilon))
    a3 = R_out / (x_out + tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64) * _stabilizer)
    print(a1.shape, a2.shape, a3.shape)
    r_in = a1*a2*a3
    print(r_in.shape)
    print()

    return r_in, None


def heuristic_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon) -> Tuple[tf.Tensor, tf.Tensor]:
    
    # calculate what we need
    x_dash = x - moving_mu 
    x_dash_dash = x_dash * (gamma / tf.sqrt(moving_var + epsilon))

    R_out_div_x_dash_dash = R_out / (x_dash_dash + beta + _stabilizer)
    R_x_dash = x_dash_dash * R_out_div_x_dash_dash
    #R_x_dash = tf.math.multiply_no_nan(x_dash_dash, R_out_div_x_dash_dash)  # R_x_dash==R_x_dash_dash
    R_b = beta * R_out_div_x_dash_dash
    
    R_x_dash_div_x_dash = R_x_dash / (x_dash + _stabilizer)
    R_in = x * R_x_dash_div_x_dash
    #R_in = tf.math.multiply_no_nan(x, R_x_dash_div_x_dash)
    R_mu = -moving_mu * R_x_dash_div_x_dash  # mu ist mean_i

    R_sink = tf.reduce_sum(R_b + R_mu, axis=1)
    return R_in, R_sink
    

def omega_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon) -> Tuple[tf.Tensor, tf.Tensor]:

    # calculate what we need
    w = gamma / tf.sqrt(moving_var + epsilon)

    x_dash = x - moving_mu 
    x_dash_dash = x_dash * w

    _sign_out =  tf.cast(tf.where(x_dash_dash + beta >= 0, 1., -1.), tf.float64)
    R_out_div_x_dash_dash = R_out / (x_dash_dash + beta + _sign_out * _stabilizer)
    R_x_dash = x_dash_dash * R_out_div_x_dash_dash
    R_b = beta * R_out_div_x_dash_dash

    _sign_out =  tf.cast(tf.where(x_dash >= 0, 1., -1.), tf.float64)
    R_x_dash_div_x_dash = R_x_dash / (x_dash + _sign_out * _stabilizer)
    R_to_in_from_x_dash = x * R_x_dash_div_x_dash
    R_mu = tf.reduce_sum(-moving_mu * R_x_dash_div_x_dash, axis=-1, keepdims=True)  # mu ist mean_i
    
    R_to_in_from_mu = (x / tf.reduce_sum(x, axis=-1, keepdims=True)) * R_mu 

    R_in = R_to_in_from_x_dash + R_to_in_from_mu

    R_sink = tf.reduce_sum(R_b, axis=1)
    return R_in, R_sink


def identity_rule(R_out) -> Tuple[tf.Tensor, tf.Tensor]:
    return R_out, None


# https://link.springer.com/chapter/10.1007/978-3-030-20518-8_24
def epsilon_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon) -> Tuple[tf.Tensor, tf.Tensor]:
    
    w = (gamma / tf.sqrt(moving_var + epsilon)) #x_dash_dash / x_dash
    b = beta - (moving_mu * w)  # We used moving_mu instead of mu
    
    xwb = x * w + b
    eps_signed = _stabilizer * tf.cast(tf.where(xwb >= 0, 1., -1.), dtype=tf.float64)
    #gamma / sqrt(self.moving_var+epsilon) * (x - self.moving_mean) + beta.
    #(w * x) - (w * self.moving_mean) + beta.
    numerator = x * w + bias_factor * (b + eps_signed) 
    denominator = xwb + eps_signed

    R_in = (numerator / denominator) * R_out

    return R_in, None


# https://link.springer.com/chapter/10.1007/978-3-030-20518-8_24
def z_rule(x, gamma, beta, R_out, moving_mu, moving_var, bias_factor, epsilon):

    w = (gamma / tf.sqrt(moving_var + epsilon)) #x_dash_dash / x_dash
    b = beta - (moving_mu * w)  # We used moving_mu instead of mu
    

    xw = x * w
    xw_sign = tf.cast(tf.where(xw >= 0, 1., -1.), dtype=tf.float64)
    b_sign = tf.cast(tf.where(b >= 0, 1., -1.), dtype=tf.float64)
    bias_signed = b * xw_sign * b_sign

    numerator = xw + bias_factor * bias_signed
    denominator = xw + bias_signed

    R_in = (numerator / denominator) * R_out
    return R_in, None


