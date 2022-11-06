from typing import Tuple

import numpy as np
from .attribution_method import AttributionMethod

from src.model.model_runner import ModelRunner

class RandomAttribution(AttributionMethod):

    def __init__(self, model_runner: ModelRunner, debug: bool = False):
        self.model_runner = model_runner
        self.mode = None
        self.debug = debug

    def get_explanation(self, data) -> Tuple[np.ndarray, np.ndarray]:
        expl_shape = (data.shape[0]) + self.model_runner.model.input.shape[1:]
        pred_shape = (data.shape[0]) + self.model_runner.model.output.shape[1:]
        return np.random.rand(*expl_shape), np.random.rand(*pred_shape)

    def reset(self) -> None:
        pass