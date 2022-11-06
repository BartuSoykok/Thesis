# Source: https://keras.io/examples/vision/grad_cam/
from typing import Tuple
import numpy as np

from lime import lime_image
from src.model.tensorflow_model_runner import TensorflowModelRunner
from src.attribution.attribution_method import AttributionMethod
from .utils import Rule


class LimeWrapper(AttributionMethod):

    def __init__(
        self, model_runner: TensorflowModelRunner, 
        mode: int  = Rule.IDENTITY_RULE, 
        debug: bool = False):
        super().__init__(model_runner, debug)
        self.mode = mode
        self.explainer = lime_image.LimeImageExplainer(verbose=False)

    def get_explanation(self, data) -> Tuple[np.ndarray, np.ndarray]:

        masks = []
        pred_fun = self.model_runner.model
        predictions = pred_fun(data)        

        for i in range(data.shape[0]): 
            explanation = self.explainer.explain_instance(
                data[i], 
                pred_fun,
                labels=np.argmax(predictions[i]),
                num_samples=1000) # TODO try stuff

            scores = explanation.local_exp[explanation.top_labels[0]]
            dict_heatmap = dict(scores)
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)  # replace segments with the scores
            temp_1 = np.expand_dims(heatmap,-1)

            temp_1 = np.expand_dims(temp_1.astype(np.float32), 0)
            masks.append(temp_1)

        masks = np.vstack(masks)

        return masks, predictions.numpy()
    
    def reset(self):
        self.model_runner.reset_outputs()
        self.explainer = lime_image.LimeImageExplainer(verbose=False)
        