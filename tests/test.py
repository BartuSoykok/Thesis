import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TODO fix GPU like wth
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())

import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from src.attribution.gradient_based import GradientBased, Grad_Rule
from src.attribution.lrp import LRP, LRP_Rule
from src.common.model_runner import ModelRunner
from src.common.display import heatmap
from src.common.model_utils import build_model
from src.common.file_utils import tf_image_generator_from_directory

conf = yaml.safe_load(open("config.yml"))
img_dir = Path(conf["paths"]["image_dir"]).resolve()
results_dir = Path(conf["paths"]["results_dir"]).resolve()

image_conf = conf['image']
input_shape = (image_conf["height"], image_conf["width"], image_conf["channels"])

model_conf = conf['model']
model_name = model_conf['name']
model_weights = model_conf['weights']

display_conf = conf['display']
color_map = plt.cm.get_cmap(display_conf['colormap'])


def restore_original_image_from_array(x, data_format='channels_first'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x

model, preprocess_input, decode_predictions = build_model(model_name, model_weights, input_shape)

checkpoint_path = results_dir / "custom_model" #/ "custom_model.ckpt"

model = tf.keras.models.load_model(str(checkpoint_path))
model_runner = ModelRunner(model)

fig = plt.figure()
for i,rule in enumerate(LRP_Rule):#[LRP_Rule.IDENTITY_RULE, LRP_Rule.EPSILON_RULE]):
    grad = LRP(model_runner, rule)

    image_generator = tf_image_generator_from_directory(
        img_dir=img_dir, 
        target_shape=(input_shape[0], input_shape[1]),
        preprocess_fn=preprocess_input)
    for image_path, image in image_generator:
        print(f"Processing image {image_path.name}")
        
        explanation, prediction = grad.get_explanation(image)
        print(prediction)
        
        explanation_default = explanation[0]
        explanation_heatmap = heatmap(explanation[0], cmap=color_map)
        explanation_restored = restore_original_image_from_array(explanation[0])

        fig.add_subplot(101+10*len(LRP_Rule)+i)

        plt.title(str(rule))
        plt.imshow(explanation_heatmap)
plt.show()
