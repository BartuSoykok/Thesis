# https://github.com/chihkuanyeh/saliency_evaluation/blob/8eb095575cf5502290a5a32e27163d1aca224580/infid_sen_utils.py#L37
# https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf
import math
import numpy as np
from skimage.segmentation import slic

from typing import List, Tuple
from src.attribution.attribution_method import AttributionMethod
from src.evaluation.evaluation_method import EvaluationMethod


class RemoveBestSuperpixel(EvaluationMethod):

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
        prediction = np.vstack(prediction)

        best_pred_idx = np.max(prediction, axis=-1, keepdims=True) == prediction
        best_pred = prediction[best_pred_idx]
        
        #expl = np.expand_dims(np.sum(expl, axis=-1), axis=-1)

        norm_prediction = np.linalg.norm(best_pred)
        max_diff = 0  # TODO whats the startng number
        
        temp_expl = expl.copy()
        temp_X = X.copy()

        results = []

        segments = slic(X, n_segments=self.sen_N, sigma = 5)
        segments = np.expand_dims(segments, -1)
        segments = np.repeat(segments, expl.shape[-1], axis=3)

        seg_rel = {}
        for i in range(np.min(segments), np.max(segments)+1):
            mask = segments == i
            rel = np.sum(expl[mask])
            seg_rel[i] = rel
        
        sorted_seg_ids = reversed(dict(sorted(seg_rel.items(), key=lambda item: item[1])).keys())

        for i in sorted_seg_ids:

            mask = segments == i
            temp_X = np.where(mask, 0, temp_X)
            segments = np.where(mask, 0, segments)
            temp_expl = np.where(mask, 0, temp_expl)
    
            
            
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
