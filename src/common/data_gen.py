
import random
from PIL import Image

import numpy as np
import tensorflow as tf
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from pathlib import Path

#from src.common.imagenet_classes import IMAGENET_CLASSES
IMAGENET_CLASSES = {"car": 0, "cat":1}

class CustomDataGen:
    
    def __init__(self,
                 num_of_samples,
                 num_of_classes,
                 input_size):

        self.input_size = input_size
        
        self.dataset = foz.load_zoo_dataset(
            "imagenet-2012",
            classes=["sports car", "sport car", "tabby", "tabby cat"],
            max_samples=num_of_samples
        )
        #self.dataset = self.dataset.filter_labels(
        #    "ground_truth", 
        #    (F("label") == "sports car") | (F("label") == "sport car") | (F("label") == "tabby") | (F("label") == "tabby cat")
        #)
        self.onehot_encoder = tf.eye(num_of_classes)

    def get_data(self):
        for sample in self.dataset:
            sample = sample.to_dict()

            ground_truth = sample['ground_truth']

            if ground_truth['label'] in ["sports car", "sport car"]:
                label = "car"
            elif ground_truth['label'] in ["tabby", "tabby cat"]:
                label = "cat"
            else:
                label = None

            if label is not None:

                img = tf.keras.preprocessing.image.load_img(sample['filepath'], target_size=self.input_size)
                img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float64)
                #img = tf.expand_dims(img, axis=0)

                label_index = IMAGENET_CLASSES[label]
                onehot_label = self.onehot_encoder[label_index]
                #onehot_label = tf.expand_dims(onehot_label, 0)
                yield img, onehot_label


class _CustomDataGen:
    
    def __init__(self,
                 num_of_samples,
                 num_of_classes,
                 input_size):

        self.input_size = input_size
        
        self.dataset = foz.load_zoo_dataset(
            #"imagenet-sample",
            "coco-2017",
            classes=list(IMAGENET_CLASSES.keys()),
            label_types=["segmentations"],
            max_samples=num_of_samples
        )
        self.onehot_encoder = tf.eye(num_of_classes)

    def get_data(self):
        for sample in self.dataset:
            sample = sample.to_dict()

            full_img = Image.open(sample["filepath"])
            metadata = sample["metadata"]
            if len(np.array(full_img).shape) == 3 and "ground_truth" in sample and sample["ground_truth"] and sample["ground_truth"]["detections"]:
                for detection in sample["ground_truth"]["detections"]:
                    if detection["label"] in IMAGENET_CLASSES:
                        bbox = detection["bounding_box"]
                        img = full_img.crop((
                            int(metadata["width"]  * detection["bounding_box"][0]),
                            int(metadata["height"] * detection["bounding_box"][1]), 
                            int(metadata["width"]  * (detection["bounding_box"][0] + detection["bounding_box"][2])), 
                            int(metadata["height"] * (detection["bounding_box"][1] + detection["bounding_box"][3]))))
                            
                        img = tf.image.convert_image_dtype(img, tf.float64)
                        img = tf.image.resize_with_pad(img, 
                            target_height=self.input_size[0], target_width=self.input_size[1])
                        #img = tf.expand_dims(img, 0)

                        label_index = IMAGENET_CLASSES[detection["label"]]
                        onehot_label = self.onehot_encoder[label_index]
                        #onehot_label = tf.expand_dims(onehot_label, 0)
                        yield img, onehot_label













class CustomImageGenFromFolder:
    
    def __init__(self,
                 folder_path: Path,
                 input_size,
                 preprocess_fn=None,
                 num_of_samples=None):

        self.folder_path = folder_path
        self.input_size = input_size
        self.preprocess_fn = preprocess_fn
        
        self.data_list = list(self.folder_path.iterdir())[:num_of_samples]
        random.shuffle(self.data_list)
                

    def get_data(self):
        for image_path in self.data_list:
            
            x = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(self.input_size[0], self.input_size[1]))
            if self.preprocess_fn is not None:
                x = self.preprocess_fn(x)
            x = tf.keras.preprocessing.image.img_to_array(x, dtype=np.float64)

            x = np.expand_dims(x, axis=0)

            yield image_path, x












class CustomDataGenFromTrainFolder:
    
    def __init__(self,
                 folder_path: Path,
                 input_size, 
                 num_of_samples=None):

        self.folder_path = folder_path
        self.num_of_samples = num_of_samples
        self.input_size = input_size
        
        self.class_paths = list(self.folder_path.iterdir())
        self.onehot_encoder = tf.eye(len(self.class_paths))

        self.classes = {
            cls_path.name: self.onehot_encoder[i].numpy()
            for i, cls_path in enumerate(self.class_paths)}
    
        self.data_list = []
        for i, cls_path in enumerate(self.class_paths):
            for image_path in list(cls_path.rglob("*.jpg"))[:self.num_of_samples]:
                self.data_list.append((i, image_path))
        random.shuffle(self.data_list)
                

    def get_data(self):
        for i, image_path in self.data_list:
            label = self.onehot_encoder[i]

            x = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(self.input_size[0], self.input_size[1]))
            img = tf.keras.preprocessing.image.img_to_array(x, dtype=np.float64)

            #img = Image.open(image_path).convert('RGB')
            #img = tf.image.convert_image_dtype(img, tf.float64)
            #img = tf.image.resize_with_pad(img, 
            #    target_height=self.input_size[0], target_width=self.input_size[1])
            
            yield img, label