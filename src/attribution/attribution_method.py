from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.model.model_runner import ModelRunner

class AttributionMethod(ABC):

    def __init__(self, model_runner: ModelRunner, debug: bool = False):
        self.model_runner = model_runner
        self.mode = None
        self.debug = debug

    @abstractmethod
    def get_explanation(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass