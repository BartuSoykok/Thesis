#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image

from scipy import ndimage
from pathlib import Path
from src.common.data_gen import CustomImageGenFromFolder

from src.common.display import heatmap, project
from src.model.model_utils import build_model
from src.model.tensorflow_model_runner import TensorflowModelRunner
from src.attribution import ATTRIBUTION_METHOD_DICT


norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

def foo(plt, image, explanation_default, color_map=None):

    fig, ax = plt.subplots()

    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('1')
    
    sp=ax.imshow(
        explanation_default, cmap=color_map,
        vmin=np.min(explanation_default), vmax=np.max(explanation_default))
    ax.set_title('heatmap')
    #ax.axis('off')
    #fig.colorbar(sp)
    return fig

def foo2(plt, image, explanation_default, color_map=None):
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(np.uint8(image), vmin=0, vmax=255)
    axs[0].set_title('original')
    axs[0].axis('off')

    sp = axs[1].imshow(
        explanation_default, cmap=color_map,
        vmin=np.min(explanation_default), vmax=np.max(explanation_default))
    axs[1].set_title('explanation')
    axs[1].axis('off')

    #fig.colorbar(sp)
    return fig



def layer_wise_relevance_propagation(conf):

    img_dir = Path(conf["paths"]["image_dir"]).resolve()
    results_dir = Path(conf["paths"]["results_dir"]).resolve()

    image_conf = conf['image']
    input_shape = (image_conf["height"], image_conf["width"], image_conf["channels"])

    display_conf = conf['display']
    color_map = plt.cm.get_cmap(display_conf['colormap'])

    model_confs = conf['models']
    attribution_method_confs = conf['attribution_methods']


    # Attribution Method
    attribution_methods = []
    for attribution_method_conf in attribution_method_confs:
        attribution_method_name = attribution_method_conf['name']
        attribution_method, modes = ATTRIBUTION_METHOD_DICT[attribution_method_name]
        
        if modes is None:
            modes = [None]
        elif "modes" in attribution_method_conf:
            modes = [modes[x] for x in attribution_method_conf['modes']]
        else:
            modes = list(modes)

        for mode in modes:
            attribution_methods.append((attribution_method, mode))





    image_generator = CustomImageGenFromFolder(
        folder_path=img_dir, 
        input_size=(input_shape[0], input_shape[1]),
        #preprocess_fn=preprocess_input
    )
 



    for model_conf in model_confs:
        model_name = model_conf['name']
        model_weights = model_conf['weights']
        model, preprocess_input, decode_predictions = build_model(model_name, model_weights, input_shape)
        print(f"{model_name} model loaded with weights {model_weights}.")
        
        model_runner = TensorflowModelRunner(model)


        for AttribMethod, mode in attribution_methods:
            if mode is None:    
                attribution_method = AttribMethod(model_runner)
            else:
                attribution_method = AttribMethod(model_runner, mode=mode)

            for image_path, image in image_generator.get_data():
                print(f"Processing image {image_path.name}")

                x = preprocess_input(image.copy())
                
                explanation, prediction = attribution_method.get_explanation(x)
                #print(decode_predictions(prediction))
                if len(explanation.shape) == 3:
                    explanation = np.expand_dims(explanation, -1)
                    
                explanation = np.expand_dims(explanation.sum(-1), -1)
                
                #explanation = explanation / np.max(np.abs(explanation), axis=(1,2,3))
                #explanation = (explanation - np.min(np.abs(explanation), axis=(1,2,3))) / (np.max(np.abs(explanation), axis=(1,2,3)) - np.min(np.abs(explanation), axis=(1,2,3)))


                if model_name == "custom":
                    image_name = f"heatmap_{mode}_{Path(model_weights).name}_{image_path.name}"
                else:
                    image_name = f"heatmap_{mode}_{model_name}_{image_path.name}"
                
                tmp = project(explanation[0], output_range=(-1, 1)).astype(np.int64)
                green = np.zeros_like(image[0], np.uint8)
                green[:,:] = (0,255,0)
                red = np.zeros_like(image[0], np.uint8)
                red[:,:] = (255,0,0)
                tmpplus = tmp.copy()
                tmpplus[tmpplus < 0] = 0
                tmpminus = tmp.copy()
                tmpminus[0 < tmpminus] = 0

                colored_overlay = tmpplus*green+tmpminus*red
                
                a = np.dot(image[0][...,:3], [0.2989, 0.5870, 0.1140])
                a = np.repeat(np.expand_dims(a, -1), 3, -1)
                print(a.shape)
                explanation_heatmap = cv2.addWeighted(
                    a.astype(np.uint8), 1,
                    colored_overlay.astype(np.uint8), 1, 0)
                    
                    
                matplotlib.image.imsave(
                    str(results_dir / attribution_method.__class__.__name__ / f"green_{image_name}"), 
                    explanation_heatmap)
                print(str(results_dir / attribution_method.__class__.__name__ / f"green_{image_name}"), "saved")

                explanation_heatmap = heatmap(explanation[0], cmap=color_map)

                matplotlib.image.imsave(
                    str(results_dir / attribution_method.__class__.__name__ / image_name), 
                    explanation_heatmap)
                print(str(results_dir / attribution_method.__class__.__name__ / image_name), "saved")


def main():
    conf = yaml.safe_load(open("attribution_config.yml"))
    layer_wise_relevance_propagation(conf)


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
