# https://github.com/chihkuanyeh/saliency_evaluation/blob/8eb095575cf5502290a5a32e27163d1aca224580/infid_sen_utils.py#L37
# https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf
import math
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from src.attribution.attribution_method import AttributionMethod
from src.evaluation.evaluation_method import EvaluationMethod


kernel = np.zeros((10,10,1), dtype=np.double)

class BlurBestPixel(EvaluationMethod):

    def __init__(self, 
        attribution_method: AttributionMethod,
        sen_N: int,
        debug: bool = False
        ):
        super().__init__(attribution_method, debug)

        self.sen_N = sen_N
        

    def get_evaluation(self, X: np.ndarray) -> Tuple[float, List[Tuple[int, float]]]:
        
        expl = []
        prediction = []
        for _batch in range(math.ceil(len(X) / self.batch_size)):
            _expl, _prediction = self.attribution_method.get_explanation(X[_batch * self.batch_size : (_batch + 1) * self.batch_size])
            self.attribution_method.reset()
            expl.append(_expl)
            prediction.append(_prediction)
        expl = np.vstack(expl)
        expl = np.expand_dims(expl.sum(-1), -1)
        
        prediction = np.vstack(prediction)

        best_pred_idx = np.max(prediction, axis=-1, keepdims=True) == prediction
        best_pred = prediction[best_pred_idx]
        
        expl = np.expand_dims(np.sum(expl, axis=-1), axis=-1)
        
        temp_expl = expl.copy()
        temp_X = X.copy()

        results = []
        
        for n in range(self.sen_N):
            mask = temp_expl == np.max(temp_expl, axis=(1,2,3), keepdims=True)
            
            
            itemindex = np.array(np.where(mask))
            _, indices = np.unique(itemindex[0], return_index=True)
            mask = np.zeros_like(mask)
            mask[itemindex[0,indices],itemindex[1,indices],itemindex[2,indices],itemindex[3,indices]] = 1

            
            
            temp_X_dash = mask.astype(np.float64)

            temp_X_dash = tf.nn.dilation2d(temp_X_dash,kernel, # TODO reuse
                (1, 1, 1, 1),"SAME","NHWC",(1, 1, 1, 1))

            temp_X_dash_dash = np.concatenate([temp_X_dash for _ in range(3)], axis=-1)
            blur_val = np.mean(temp_X * temp_X_dash_dash, axis=(1,2,3), keepdims=True)
            
            temp_expl = np.where(temp_X_dash, 0, temp_expl)
            temp_X = np.where(temp_X_dash, blur_val, temp_X)

            
            prediction_eps = []
            for _batch in range(math.ceil(len(temp_X) / self.batch_size)):
                _prediction = self.attribution_method.model_runner.model(
                    temp_X[_batch * self.batch_size : (_batch + 1) * self.batch_size])
                self.attribution_method.reset()
                prediction_eps.append(_prediction)
            prediction_eps = np.vstack(prediction_eps)
            best_pred_eps = prediction_eps[best_pred_idx]

            new_diff = best_pred_eps
            results.append(new_diff)
        results = np.vstack(results).transpose()
        max_diff = np.mean(np.trapz(results, axis=1))
        
        display_results = [
            np.mean( np.trapz(results[:, :i]) )
            for i in range(1, results.shape[1]+1)
        ]
        
        return max_diff, display_results
