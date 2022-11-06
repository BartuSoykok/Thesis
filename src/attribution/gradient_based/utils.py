from enum import Enum

class Rule(Enum):
    GRAD_X_INPUT = 1
    GRAD_ONLY = 2
    INTEGRATED_GRAD = 3
    RANDOM_BASELINE_INTEGRATED_GRAD = 4
    