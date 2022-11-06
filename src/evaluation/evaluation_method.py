from abc import ABC, abstractmethod
from typing import List, Tuple

from src.attribution.attribution_method import AttributionMethod

class EvaluationMethod(ABC):

    def __init__(self, attribution_method: AttributionMethod, debug: bool = False):
        self.attribution_method = attribution_method
        self.debug = debug
        self.batch_size = 5

    @abstractmethod
    def get_evaluation(self, data) -> Tuple[float, List[Tuple[int, float]]]: # TODO maybe Any
        pass