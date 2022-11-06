from .sensitivity_n import SensitivityN
from .remove_best_pixel import RemoveBestPixel
from .remove_best_superpixel import RemoveBestSuperpixel
from .blur_best_pixel import BlurBestPixel
from .blur_best_superpixel import BlurBestSuperpixel
from .irof_remove import IROF_Remove
from .irof_blur import IROF_Blur


EVALUATION_METHOD_DICT = {
    "SensitivityN": SensitivityN,
    "RemoveBestPixel": RemoveBestPixel,
    "RemoveBestSuperpixel": RemoveBestSuperpixel,
    "BlurBestPixel": BlurBestPixel,
    "BlurBestSuperpixel": BlurBestSuperpixel,
    "IROF": IROF_Remove,
    "IROF_Blur": IROF_Blur
}
