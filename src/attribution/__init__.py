from .random_attribution import RandomAttribution
from .gradient_based import GradientBased, Grad_Rule
from .lime_wrapper import LimeWrapper, LIME_Rule
from .lrp import LRP, LRP_Rule


ATTRIBUTION_METHOD_DICT = {
    "RandomAttribution": (RandomAttribution, None),
    "GradientBased": (GradientBased, Grad_Rule),
    "LIME": (LimeWrapper, LIME_Rule),
    "LRP": (LRP, LRP_Rule)
}