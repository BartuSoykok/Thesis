# https://github.com/chihkuanyeh/saliency_evaluation/blob/8eb095575cf5502290a5a32e27163d1aca224580/infid_sen_utils.py#L37
# https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf
import math
from typing import List, Tuple
import numpy as np
from scipy import stats
#import tensorflow as tf
from src.attribution.attribution_method import AttributionMethod
from src.evaluation.evaluation_method import EvaluationMethod


class SensitivityN(EvaluationMethod):

    def __init__(self, 
        attribution_method: AttributionMethod, 
        sen_N: int,
        sen_r: float,
        debug: bool = False
        ):
        super().__init__(attribution_method, debug)
        
        self.sen_N = sen_N
        self.sen_r = sen_r

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
        sample =  np.random.uniform(size=X.shape)
        sum_prev = np.sum(expl, axis=(1,2,3))

        temp_expl = expl.copy()
        temp_X = X.copy()

        att_list = []
        pred_list = []

        for n in range(self.sen_N):
            

            a = temp_expl.reshape(temp_expl.shape[0], -1)
            a = np.sort(a)[:, (-1*self.sen_r):]
            
            #mask = temp_expl == np.max(, axis=(1,2,3), keepdims=True)
            mask = np.array([np.in1d(temp_expl[a_idx], a[a_idx]) for a_idx in range(a.shape[0])])
            mask = mask.reshape(temp_expl.shape)

            temp_expl = np.where(mask, 0, temp_expl)
            removed_att = sum_prev - temp_expl.sum(axis=(1,2,3))#temp_expl[mask].sum(axis=(1,2,3))
            temp_X = np.where(mask, sample, temp_X)

            prediction_eps = []
            for _batch in range(math.ceil(len(temp_X) / self.batch_size)):
                _prediction = self.attribution_method.model_runner.model(
                    temp_X[_batch * self.batch_size : (_batch + 1) * self.batch_size])
                prediction_eps.append(_prediction)
            prediction_eps = np.vstack(prediction_eps)
            best_pred_eps = prediction_eps[best_pred_idx]

            att_list.append(removed_att)
            pred_list.append(best_pred - best_pred_eps)

        att_list = np.vstack(att_list)
        pred_list = np.vstack(pred_list)

        steps = [
            np.mean([
                stats.pearsonr(att_list[:n,m], pred_list[:n,m])[0]
                for m in range(att_list.shape[-1])
            ])
            for n in range(3,self.sen_N)
        ]
        return steps[-1], steps 
    
# TODO usage of these / importance [Waiting]
# TODO Goal: Imperically valid. [Waiting]
# TODO How do you evaluate lime (Oh god!) [Waiting]
