# https://keras.io/examples/vision/integrated_gradients/

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import pandas as pd
import yaml
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from pathlib import Path

from src.model.tensorflow_model_runner import TensorflowModelRunner
from src.model.model_utils import build_model
from src.common.data_gen import CustomDataGenFromTrainFolder
from src.attribution import *
from src.evaluation import *


num_of_samples = 10

def layer_wise_relevance_propagation(conf):

    img_dir = Path(conf["paths"]["image_dir"]).resolve()
    results_dir = Path(conf["paths"]["results_dir"]).resolve()
    csv_results_dir = Path(conf["paths"]["csv_results_dir"]).resolve()

    image_conf = conf['image']
    input_shape = (image_conf["height"], image_conf["width"], image_conf["channels"])

    model_confs = conf['models']
    attribution_method_confs = conf['attribution_methods']
    evaluation_method_confs = conf['evaluation_methods']
    

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

    eval_methods = []
    for evaluation_method_conf in evaluation_method_confs:
        evaluation_method_name = evaluation_method_conf['evaluation_method']
        evaluation_method_params = evaluation_method_conf['params']
        eval_method = EVALUATION_METHOD_DICT[evaluation_method_name]
        
        eval_methods.append((eval_method, evaluation_method_params))    




    image_generator = CustomDataGenFromTrainFolder(
        folder_path=img_dir, 
        input_size=(input_shape[0], input_shape[1]),
        num_of_samples=num_of_samples
        #preprocess_fn=preprocess_input
    )
    data = np.stack([x for x, y in image_generator.get_data()])







    eval_header = ['Model', 'EvalMethod', "EvalParam", 'AtribMethod', 'n', 'Result', 'FinalScore']
    output_file = csv_results_dir / "output_file.csv"
    df = pd.DataFrame(columns=eval_header)
    df.to_csv(output_file, mode='w', index=False)

    for model_conf in model_confs:
        model_name = model_conf['name']
        model_weights = model_conf['weights']
        model, preprocess_input, decode_predictions = build_model(model_name, model_weights, input_shape)
        print(f"{model_name} model loaded with weights {model_weights}.")
        
        model_runner = TensorflowModelRunner(model)
        data = preprocess_input(data)

        for AttribMethod, mode in attribution_methods:
            if mode is None:    
                attribution_method = AttribMethod(model_runner)
            else:
                attribution_method = AttribMethod(model_runner, mode=mode)

            for EvalMethod, params in eval_methods:
                eval_method = EvalMethod(attribution_method, **params)
                
                
                eval_start_time = time.time()
                eval_score, results = eval_method.get_evaluation(data)
                eval_tot_time = time.time() - eval_start_time

                if model_name == "custom":
                    model_display_name = f"{model_name}_{Path(model_weights).name}"
                else:
                    model_display_name = model_name
                eval_name = f"{type(eval_method).__name__}"
                atrib_name = f"{type(attribution_method).__name__}({attribution_method.mode})"
            
                print(f"{eval_name} - {atrib_name}:\t{eval_score:0.8f} (Time took {eval_tot_time:0.2f}s)")
                
                result_list = [
                    [model_display_name, eval_name, params, atrib_name, i, result, eval_score]
                    for i, result in enumerate(results)
                ]
                df = pd.DataFrame(result_list, columns=eval_header)
                df.to_csv(output_file, mode='a', index=False, header=False)
                

def main():
    conf = yaml.safe_load(open("evaluation_config.yml"))
    layer_wise_relevance_propagation(conf)


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

