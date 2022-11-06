from typing import Optional, Tuple
from pathlib import Path

from skimage.io import imread
from skimage.transform import resize
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image


def image_generator_from_directory(
    img_dir: Path,
    target_shape: Optional[Tuple[int, int]] = None,
    preprocess_fn=None
    ) -> Tuple[Path, np.ndarray]:
    for image_path in img_dir.iterdir():
        
        """image = cv2.imread(str(image_path))
        if target_shape:
            image = cv2.resize(image, target_shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)"""
        image = imread(str(image_path))
        image = resize(image, target_shape) #* 255
        
        if preprocess_fn:
            image = preprocess_fn(image)

        yield image_path, image

def tf_image_generator_from_directory(
    img_dir: Path,
    target_shape: Optional[Tuple[int, int]] = None,
    preprocess_fn=None
    ) -> Tuple[Path, tf.Tensor]:
    for image_path in img_dir.iterdir():
        
        img = image.load_img(image_path, target_size=target_shape)
        img = image.img_to_array(img, dtype=np.float64)
        img = tf.expand_dims(img, axis=0)
        
        if preprocess_fn:
            img = preprocess_fn(img)

        yield image_path, img