import numpy as np
import tensorflow as tf

from src.model.model_runner import ModelRunner


class TensorflowModelRunner(ModelRunner):

    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug

        self.layer_outputs = dict()

    def reset_outputs(self):
        self.layer_outputs = dict()

    def predict(self, image):

        input_layer = self.model._input_layers[0]

        out = input_layer(image)
        self.layer_outputs = {input_layer.name: tf.cast(out, tf.float64)}

        outbound_layer = self.get_outbound_layers(input_layer)
        self._iterate_layer(outbound_layer)
        
        output = self.layer_outputs[self.model._output_layers[0].name]
        return output.numpy()
        
    def _is_level_reached(self, layer):
        # Missing outbound layer relevance
        for inbound_layer in self.get_inbound_layers(layer): 
            if inbound_layer.name not in self.layer_outputs:
                return False
        return True

    def _iterate_layer(self, layers, level=0):
        recursion_stack = [(layer, level) for layer in layers]

        while recursion_stack:
            #print(len(recursion_stack), [L.name for L,_ in recursion_stack])
            current_layer, currect_level = recursion_stack.pop()

            if current_layer.name not in self.layer_outputs and self._is_level_reached(current_layer):
                
                # Get inbound layers
                inbound_layers = self.get_inbound_layers(current_layer)
                if len(inbound_layers) == 1:
                    inputs = self.layer_outputs[inbound_layers[0].name]
                else:
                    inputs = [self.layer_outputs[inbound_layer.name] for inbound_layer in inbound_layers]

                # Run layer
                out = current_layer(inputs)
                self.layer_outputs[current_layer.name] = tf.cast(out, tf.float64)
            
                # Recurse
                for outbound_layer in self.get_outbound_layers(current_layer):
                    recursion_stack.append((outbound_layer, currect_level+1))

    def _recurse_layer(self, layer, level=0):
        """
        Not used anymore iterative should be more performant
        """
        if layer.name in self.layer_outputs:
            return
        #print(f"{level:5} -> {layer.name}")

        # Get inbound layers
        inbound_layers = self.get_inbound_layers(layer)
        if len(inbound_layers) == 1:
            inputs = self.layer_outputs[inbound_layers[0].name]
        else:
            inputs = [self.layer_outputs[inbound_layer.name] for inbound_layer in inbound_layers]

        # Run layer
        out = layer(inputs)
        self.layer_outputs[layer.name] = out
        
        # Get outbound layers
        outbound_layers = self.get_outbound_layers(layer)
        
        # Recurse
        for outbound_layer in outbound_layers:
            self._recurse_layer(outbound_layer, level+1)

    def get_output_layer(self):
        return self.model._output_layers[0]  # TODO multiple next

    def create_masked_output_with_max_values(self, output_layer_name: str): # TODO optimize?        
        lrp_prediction = self.layer_outputs[output_layer_name]
        target_index = [(i, max_index) for i, max_index in enumerate(np.argmax(lrp_prediction, 1))]
        masked_output = self.get_masked_output(lrp_prediction, target_index)
        return masked_output

    @staticmethod
    def get_masked_output(x, indices, dtype=tf.float64):
        zeros = np.zeros_like(x)
        for index in indices:
            zeros[index] = x[index]
        return tf.convert_to_tensor(zeros, dtype=dtype)

    @staticmethod 
    def get_inbound_layers(layer):
        inputs = []
        for inbound_node in layer._inbound_nodes:
            inbound_layers = inbound_node.inbound_layers
            if type(inbound_layers) is list: 
                inputs.extend(inbound_layers)
            else:
                inputs.append(inbound_layers)
        return inputs

    @staticmethod 
    def get_outbound_layers(layer):
        outputs = []
        for outbound_node in layer._outbound_nodes:
            outbound_layer = outbound_node.outbound_layer
            if type(outbound_layer) is list: 
                outputs.extend(outbound_layer)
            else:
                outputs.append(outbound_layer)
        return outputs
