import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib/"

# TensorFlow and tf.keras
import tensorflow as tf

from src.common.data_gen import CustomDataGenFromTrainFolder

print(tf.__version__)

# Helper libraries
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.model_utils import build_model
from src.common.display import heatmap
from src.model.tensorflow_model_runner import TensorflowModelRunner


epochs = 8
batch_size = 32
period = 1
num_of_classes = 2


conf = yaml.safe_load(open("train_config.yml"))

eval_image_dir = Path(conf["paths"]["eval_image_dir"]).resolve()
train_image_dir = Path(conf["paths"]["train_image_dir"]).resolve()
results_dir = Path(conf["paths"]["results_dir"]).resolve()
csv_results_dir = Path(conf["paths"]["csv_results_dir"]).resolve()

image_conf = conf['image']
input_shape = (image_conf["height"], image_conf["width"], image_conf["channels"])




def foo1():
    resnet_pre = tf.keras.applications.resnet50.ResNet50(
        input_shape=input_shape, include_top=False, 
        pooling='max', weights='imagenet')
        
    resnet_pre.trainable = True
    
    x = tf.keras.layers.Dense(num_of_classes, activation='softmax')(resnet_pre.output)

    return tf.keras.Model(resnet_pre.input, x), tf.keras.applications.resnet50.preprocess_input
def foo2():
    resnet_pre = tf.keras.applications.VGG19(
        input_shape=input_shape, include_top=False, 
        weights='imagenet')
        
    for layer in resnet_pre.layers:
        layer.trainable = True

    x = tf.keras.layers.GlobalMaxPooling2D()(resnet_pre.output)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(num_of_classes, activation='softmax')(x)

    return tf.keras.Model(resnet_pre.input, x), tf.keras.applications.vgg19.preprocess_input

model, preprocess_input = foo1()
model.summary()



datagen = CustomDataGenFromTrainFolder(train_image_dir, input_shape)
dataset = tf.data.Dataset.from_generator(
    datagen.get_data,
    output_signature=(
        tf.TensorSpec(list(input_shape), tf.float64),
        tf.TensorSpec([num_of_classes], tf.float64))
    )
dataset = dataset.map(lambda x, y: (preprocess_input(x), y))
#dataset = dataset.map(lambda x, y: (x / 255.0, y))


train_dataset = dataset.batch(batch_size)

datagen = CustomDataGenFromTrainFolder(eval_image_dir, input_shape)
dataset = tf.data.Dataset.from_generator(
    datagen.get_data,
    output_signature=(
        tf.TensorSpec(list(input_shape), tf.float32),
        tf.TensorSpec([num_of_classes], tf.float32))
    )
dataset = dataset.map(lambda x, y: (preprocess_input(x), y))
#dataset = dataset.map(lambda x, y: (x / 255.0, y))

test_dataset = dataset.batch(batch_size)

test_images = []
test_labels = []
for x, y in test_dataset:
    #print(x.shape, y[0])
    #plt.imshow(x[0].numpy())
    #plt.show()
    #plt.savefig(str(csv_results_dir / f"test.jpg"), bbox_inches='tight')
    #exit()
    test_images.append(x)
    test_labels.append(y)
    
test_images = tf.concat(test_images, axis=0)
test_labels = tf.concat(test_labels, axis=0)
print(test_images.shape,test_labels.shape)

print(test_images.shape, test_labels.shape)




save_path = results_dir / "saved_model"
checkpoint_path = results_dir / "custom_model_{epoch:02d}" #/ "custom_model.ckpt"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=(checkpoint_path),
    save_freq='epoch',
    period=period)


if save_path.exists():
    model = tf.keras.models.load_model(str(save_path))
else:
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[model_checkpoint_callback])

    model.save(str(save_path))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(csv_results_dir / f"accuracy.jpg"), bbox_inches='tight')
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(csv_results_dir / f"loss.jpg"), bbox_inches='tight')
    plt.clf()


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

