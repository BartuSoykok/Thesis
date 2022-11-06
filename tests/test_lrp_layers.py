import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TODO fix GPU like wth

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tensorflow as tf
import numpy as np

from src.common.model_runner import ModelRunner
from src.attribution.lrp import LRP, LRP_Rule
from .layer_fixtures import *


def test_dense_model(dense_model, test_data_flattened):
    _, (X_val, Y_val) = test_data_flattened

    model_runner = ModelRunner(dense_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


@pytest.mark.parametrize("kernel", [(3,3), (5,5)])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("strides", [(1,1), 2])
@pytest.mark.parametrize("padding", ['valid', 'same'])
@pytest.mark.parametrize("kernel_initializer", [None])
def test_conv_model(generate_conv_model, test_data, 
                    kernel, use_bias, strides, padding, kernel_initializer):
    _, (X_val, Y_val) = test_data
    filters=8
    conv_model = generate_conv_model(filters, kernel, use_bias, strides, padding, 
                kernel_initializer=kernel_initializer)

    model_runner = ModelRunner(conv_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


@pytest.mark.parametrize("kernel", [(1,1)])
@pytest.mark.parametrize("depth_multiplier", [1,2])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("strides", [(1,1), (3,3)])
@pytest.mark.parametrize("padding", ['valid', 'same'])
def test_depthconv_model(generate_depthconv_model, test_data, 
                    kernel, depth_multiplier, use_bias, strides, padding):
    _, (X_val, Y_val) = test_data
    depthconv_model = generate_depthconv_model(kernel, depth_multiplier,use_bias, strides, padding)

    model_runner = ModelRunner(depthconv_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


@pytest.mark.parametrize("epsilon", [0.001, 2e-5])
@pytest.mark.parametrize("mode", LRP_Rule)
def test_bn_model(generate_bn_model, test_data, epsilon, mode):
    _, (X_val, Y_val) = test_data
    bn_model = generate_bn_model(epsilon=epsilon)

    model_runner = ModelRunner(bn_model)
    x = X_val[:7]

    assert_lrp(model_runner, x, mode)


@pytest.mark.parametrize("pool_size", [(2,2), 3])
@pytest.mark.parametrize("strides", [(1,1), 2])
@pytest.mark.parametrize("padding", ['valid', 'same'])
def test_pooling_model(generate_pooling_model, test_data,
                        pool_size, strides, padding):
    _, (X_val, Y_val) = test_data
    pooling_model = generate_pooling_model(pool_size, strides, padding)

    model_runner = ModelRunner(pooling_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


def test_global_pooling_model(global_pooling_model, test_data):
    _, (X_val, Y_val) = test_data

    model_runner = ModelRunner(global_pooling_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


@pytest.mark.parametrize("padding", [2, (2, 3), ((1,2),(3,4))])
def test_zeropadding_model(generate_zeropadding_model, test_data, padding):
    _, (X_val, Y_val) = test_data
    zeropadding_model = generate_zeropadding_model(padding)

    model_runner = ModelRunner(zeropadding_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


def test_add_model(generate_add_model, test_data):
    _, (X_val, Y_val) = test_data
    add_model = generate_add_model()

    model_runner = ModelRunner(add_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


def test_conc_model(generate_conc_model, test_data):
    _, (X_val, Y_val) = test_data
    conc_model = generate_conc_model()

    model_runner = ModelRunner(conc_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


def test_convblock_model(generate_convblock_model, test_data):
    _, (X_val, Y_val) = test_data
    add_model = generate_convblock_model()

    model_runner = ModelRunner(add_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)

    
def test_resblock_model(generate_resblock_model, test_data):
    _, (X_val, Y_val) = test_data
    add_model = generate_resblock_model()

    model_runner = ModelRunner(add_model)
    x = X_val[:7]

    assert_lrp(model_runner, x)


def assert_lrp(model_runner, x, mode=LRP_Rule.IDENTITY_RULE):
    lrp = LRP(model_runner=model_runner, mode=mode, bias_factor=1.0)

    tf_prediction = model_runner.model(x)
    custom_prediction = model_runner.predict(x)

    np.testing.assert_almost_equal(tf_prediction, custom_prediction)  # abs(desired-actual) < 1.5 * 10**(-decimal) [decimal = 7]

    relevance = lrp.relevance_propagation()
    
    R_in = tf.reduce_sum(tf.reduce_max(custom_prediction, axis=1)).numpy()
    R_out = tf.reduce_sum(relevance).numpy()
    if lrp.layer_sinks:
        sinked_layers = [tf.reduce_sum(sl) for sl in lrp.layer_sinks.values()]
        sinked_amount = tf.add_n(sinked_layers)
        sinked_amount = tf.reduce_sum(sinked_amount).numpy()
        R_out += sinked_amount
    assert np.isclose(R_in, R_out)