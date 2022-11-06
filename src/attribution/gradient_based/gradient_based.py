# Source: https://keras.io/examples/vision/grad_cam/
from typing import Tuple
import numpy as np
import tensorflow as tf

from src.model.tensorflow_model_runner import TensorflowModelRunner
from src.attribution.attribution_method import AttributionMethod
from .utils import Rule


class GradientBased(AttributionMethod):

    def __init__(self, model_runner: TensorflowModelRunner, mode: int, debug: bool = False):
        super().__init__(model_runner, debug)
        self.mode = mode

    def get_explanation(self, data) -> Tuple[np.ndarray, np.ndarray]:
            
            num_steps = 50
            num_runs = 2

            data = tf.convert_to_tensor(data)
            preds = self.model_runner.model(data)
            top_pred_idx = np.max(preds, axis=-1,keepdims=True) == preds
            #top_pred_idx = [(i, max_index) for i, max_index in enumerate(np.argmax(preds, 1))]

            if self.mode == Rule.GRAD_X_INPUT:
                return self.grad_x_input(data,
                 top_pred_idx), preds.numpy()
            if self.mode == Rule.GRAD_ONLY:
                return self.grad_only(data, 
                top_pred_idx), preds.numpy()
            elif self.mode == Rule.INTEGRATED_GRAD:
                return self.get_integrated_gradients(data, 
                top_pred_idx, baseline=None, num_steps=num_steps), preds.numpy()
            elif self.mode == Rule.RANDOM_BASELINE_INTEGRATED_GRAD:
                return self.random_baseline_integrated_gradients(data, 
                top_pred_idx, num_steps=num_steps, num_runs=num_runs), preds.numpy()
            else:
                raise NotImplementedError(f"GradientBased {self.mode} not implemented.")

    def reset(self):
        self.model_runner.reset_outputs()
        
    def grad_only(self, data, top_pred_idx) -> Tuple[np.ndarray, np.ndarray]:

        with tf.GradientTape() as tape:
            tape.watch(data)
            prediction = self.model_runner.model(data)
            probs = tf.nn.softmax(prediction, axis=-1)[top_pred_idx]

        grads = tape.gradient(probs, data)
        
        explanation = tf.abs(grads)
        
        return explanation.numpy()


    def grad_x_input(self, data, top_pred_idx) -> Tuple[np.ndarray, np.ndarray]:

        with tf.GradientTape() as tape:
            tape.watch(data)
            prediction = self.model_runner.model(data)
            probs = tf.nn.softmax(prediction, axis=-1)[top_pred_idx]

        grads = tape.gradient(probs, data)
        
        explanation = grads * data
        
        return explanation.numpy()













    def _get_gradients(self, img_input, top_pred_idx):
        
        images = tf.cast(img_input, tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = self.model_runner.model(images)
            probs = tf.nn.softmax(preds, axis=-1)[top_pred_idx]

        grads = tape.gradient(probs, images)
        return grads

    def get_integrated_gradients(self, img_input, top_pred_idx, baseline=None, num_steps=50):
        
        img_input = img_input.numpy()
        if baseline is None:
            baseline = np.zeros_like(img_input).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)

        img_input = img_input.astype(np.float32)
        a = (img_input - baseline)
        interpolated_image = [
            baseline + (step / num_steps) * a
            for step in range(num_steps + 1)
        ]
        interpolated_image = np.array(interpolated_image).astype(np.float32)

        grads = []
        for i, img in enumerate(interpolated_image):
            #img = tf.expand_dims(img, axis=0)
            grad = self._get_gradients(img, top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float64)

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        integrated_grads = (img_input - baseline) * avg_grads
        return integrated_grads.numpy()
        
    def random_baseline_integrated_gradients(self, img_input, top_pred_idx, num_steps=50, num_runs=2):
        integrated_grads = []
        
        for run in range(num_runs):
            baseline = np.random.random(img_input.shape) * 255
            igrads = self.get_integrated_gradients(
                img_input=img_input,
                top_pred_idx=top_pred_idx,
                baseline=baseline,
                num_steps=num_steps,
            )
            integrated_grads.append(igrads)

        integrated_grads = tf.convert_to_tensor(integrated_grads)
        return tf.reduce_mean(integrated_grads, axis=0).numpy()
