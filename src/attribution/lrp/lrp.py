from typing import Any
import tensorflow as tf
import numpy as np
import math

from .layers import *
from src.model.tensorflow_model_runner import TensorflowModelRunner
from src.attribution.attribution_method import AttributionMethod


class LRP(AttributionMethod):

    def __init__(self, model_runner: TensorflowModelRunner, mode: int, bias_factor: float = 0.0, debug: bool = False):
        super().__init__(model_runner, debug)
        self.mode = mode
        self.bias_factor=bias_factor

        self.eps = 1e-32 # 1e-4

        self.layer_relevances = dict()
        self.layer_sinks = dict()
        

    def reset_relevances(self):
        self.layer_relevances = dict()
        self.layer_sinks = dict()

    def reset(self):
        self.reset_relevances()
        self.model_runner.reset_outputs()

    def get_explanation(self, data) -> Tuple[np.ndarray, np.ndarray]:
        
        self.model_runner.reset_outputs()
        self.reset_relevances()
    
        prediction = self.model_runner.predict(data)
        explanation = self.relevance_propagation()

        return explanation, prediction

    def relevance_propagation(self) -> np.ndarray:
        output_layer = self.model_runner.get_output_layer()
        masked_output = self.model_runner.create_masked_output_with_max_values(output_layer.name)

        
        self.layer_relevances = {output_layer.name: masked_output}
        
        #out = self._relevance_propagation(output_layer)
        out = self._iterate_propagation(output_layer)

        #for k, v in self.layer_relevances.items():
        #    tf.print(k, tf.reduce_sum(v))

        return out.numpy()
    
    
    def _is_level_reached(self, layer):
        # Missing outbound layer relevance
        for outbound_layer in TensorflowModelRunner.get_outbound_layers(layer):
            if outbound_layer.name not in self.layer_relevances:
                return False
        return True

    def _relevance_propagation(self, layer, level=0):
        """
        Not used anymore iterative should be more performant
        """
        out = self.relprop(layer)
        
        if out is not None:
            return out

        for inbound_layer in TensorflowModelRunner.get_inbound_layers(layer):
            if self._is_level_reached(inbound_layer):
                self._relevance_propagation(inbound_layer, level+1)
            else:
                pass  # Layer not reached yet.
    
    def _iterate_propagation(self, layer, level=0):
        recursion_stack = [(layer, level)]

        while recursion_stack:
            current_layer, currect_level = recursion_stack.pop()

            if self._is_level_reached(current_layer):
                out = self.relprop(current_layer, currect_level)
                
                if out is not None:
                    return out

                for inbound_layer in TensorflowModelRunner.get_inbound_layers(current_layer):
                    if inbound_layer.name not in self.layer_relevances:
                        recursion_stack.append((inbound_layer, currect_level+1))
        

    def relprop(self, layer, level=None):

        outbound_layers = TensorflowModelRunner.get_outbound_layers(layer)
        #print([tf.reduce_sum(self.layer_relevances[asdf.name]) for asdf in outbound_layers])
        if 0 == len(outbound_layers):
            r = tf.math.add_n(self.layer_relevances.values())
        else:
            r_list = []
            for temp_layer in outbound_layers:
                #out_X = self.model_runner.layer_outputs[temp_layer.name]
                out_r = self.layer_relevances[temp_layer.name]
                
                if "Add" in str(type(temp_layer)):
                    
                    _name = f"{temp_layer.name}_to_{layer.name}"
                    temp_r = self.layer_relevances[_name]
                    r_list.append(temp_r)
                    
                elif "Concatenate" in str(type(temp_layer)):
                    axis = layer # TODO different axis
                    temp_offset, temp_length = 0, 0
                    
                    ins_of_out_layer = TensorflowModelRunner.get_inbound_layers(temp_layer)
                    for in_of_out_layer in ins_of_out_layer:
                        if layer == in_of_out_layer:
                            temp_length = self.model_runner.layer_outputs[in_of_out_layer.name].shape[-1]
                            break
                        else:
                            temp_offset += self.model_runner.layer_outputs[in_of_out_layer.name].shape[-1]
                    
                    r_list.append(out_r[..., temp_offset:temp_offset+temp_length])

                else:
                    r_list.append(out_r)
            
                #tf.print(temp_layer, tf.reduce_sum(out_r))
            #print("----------------------------------------------------------------")
            #for llll in r_list: tf.print("add_list->", tf.reduce_sum(llll))
            r = tf.math.add_n(r_list)
        
        inbound_layers = TensorflowModelRunner.get_inbound_layers(layer)
        if len(inbound_layers) == 1:
            x = self.model_runner.layer_outputs[inbound_layers[0].name]
        else:
            x = [self.model_runner.layer_outputs[inbound_layer.name]
                 for inbound_layer in inbound_layers]
                 
        if "input" in str(type(layer)):
            self.layer_relevances[layer.name] = r
            return r
        else:
            r_prev, r_sink = self.propagate_relevance_to_layer(layer, r, x, level)
            self.layer_relevances[layer.name] = r_prev
            if r_sink is not None:
                self.layer_sinks[layer.name] = r_sink
        

    def propagate_relevance_to_layer(self, layer, r, x, level=None):
        r_prev, r_sink = None, None
        x_out = self.model_runner.layer_outputs[layer.name]
        #tf.print(layer.name, tf.reduce_sum(r), r.shape)
        if "input" in str(type(layer)):
            raise Exception("Error: input layer should not be propagated.")

        elif "Flatten" in str(type(layer)) or "Reshape" in str(type(layer)):
            r_prev = propagate_flatten_layer(x, r)

        elif "Conv" in str(type(layer)):
            w = layer.weights[0]
            b = layer.bias
            strides = layer.strides
            padding = layer.padding
            activation = layer.activation
            dilation_rate = layer.dilation_rate
            #if dilation_rate!=(1, 1):
            #    tf.print("dilation_rate is:", dilation_rate)
            #if activation is not None:
            #    tf.print("Conv layer activation is:", activation)
            
            
            if b is not None: b = tf.cast(b, tf.float64)
            w = tf.cast(w, tf.float64)
                
            r_prev, r_sink = propagate_conv_layer(
                x=x, w=w, b=b, 
                x_out=x_out, r=r,
                bias_factor=self.bias_factor, eps=self.eps,
                strides=strides,
                padding=padding,
                level=level)
            #tf.print("conv r change?:", tf.reduce_sum(r)-tf.reduce_sum(r_prev))

        elif "BatchNormalization" in str(type(layer)):

            gamma, beta, moving_mean, moving_variance = None, None, None, None
            for layer_weight in layer.weights: # TODO write better code
                if "gamma": gamma = tf.cast(layer_weight, tf.float64)
                if "beta": beta = tf.cast(layer_weight, tf.float64)
                if "moving_mean": moving_mean = tf.cast(layer_weight, tf.float64)
                if "moving_variance": moving_variance = tf.cast(layer_weight, tf.float64)
            
            if gamma is None: 1
            
            r_prev, r_sink = propagate_bn_layer(
                x=x, x_out=x_out, gamma=gamma, beta=beta, 
                R_out=r, moving_mu=moving_mean,
                moving_var=moving_variance, 
                bias_factor=self.bias_factor, epsilon=self.eps,
                mode= self.mode)
            #tf.print("bn r change?:", tf.reduce_sum(r)-tf.reduce_sum(r_prev))

        elif "Padding" in str(type(layer)):
            padding = layer.padding

            start_r = tf.reduce_sum(r)
            r_prev = remove_padding(r, padding)
            r_sink = start_r - tf.reduce_sum(r_prev)
        
        elif "Global" in str(type(layer)) and "Pool" in str(type(layer)):
            r_prev, r_sink = propagate_global_pooling_layer(
                x=x, x_out=x_out, r=r)

        elif "Pooling" in str(type(layer)):
            pool_size = layer.pool_size
            strides = layer.strides
            padding = layer.padding
            
            r_prev, r_sink = propagate_pooling_layer(
                x=x, pool_size=pool_size, 
                x_out=x_out, r=r, strides=strides, padding=padding)

        elif "Dense" in str(type(layer)):
            w = layer.weights[0]
            b = layer.bias
            activation = layer.activation
            #if activation is not None:
            #    tf.print("Dense layer activation is:", activation)
            
            if b is not None: b = tf.cast(b, tf.float64)
            w = tf.cast(w, tf.float64)

            r_prev = propagate_dense_layer(
                x=x, w=w, b=b, 
                x_out=x_out, r=r,
                bias_factor=self.bias_factor, eps=self.eps)
                
        elif "Add" in str(type(layer)):
            ins_layer = TensorflowModelRunner.get_inbound_layers(layer)
            sign_out = tf.cast(tf.where(x_out >= 0, 1., -1.), tf.float64)
            denom = x_out + self.eps * sign_out
            for in_layer in ins_layer:
                cur_x = self.model_runner.layer_outputs[in_layer.name]
                #temp_ratio = (cur_x + self.eps * sign_out) / denom
                temp_ratio = 1 / len(ins_layer)
                temp_r = temp_ratio * r

                _name = f"{layer.name}_to_{in_layer.name}"
                self.layer_relevances[_name] = temp_r
            r_prev = r
                
        else:
            #tf.print("Warning unknown layer:", layer)
            r_prev = r
            
            # TODO learn relu check papers

        #r_prev = r_prev / tf.reduce_sum(r_prev, axis=tuple(range(1, len(r_prev.shape))))
        #a = tf.reduce_sum(r_prev)
        #b = 0 if r_sink is None else tf.reduce_sum(r_sink)
        #tf.print(layer.name, a + b, f"{a}+{b}")
        #print()
        
        return r_prev, r_sink


