from pathlib import Path
import tensorflow as tf
import matplotlib
import numpy as np
import innvestigate
tf.compat.v1.disable_eager_execution()

model = tf.keras.applications.ResNet50()
preprocess = tf.keras.applications.resnet.preprocess_input

model = innvestigate.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer("deep_taylor", model)

file_path = Path("data/images/4.jpg").resolve() 
x = tf.keras.preprocessing.image.load_img(
    str(file_path), target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(x, dtype=np.float64)
image = np.expand_dims(image, 0)

x = preprocess(image)
# Apply analyzer w.r.t. maximum activated output-neuron
a = analyzer.analyze(x)

# Aggregate along color channels and normalize to [-1, 1]
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))
# Plot
matplotlib.image.imsave(
    str(Path("data/results/4.jpg").resolve() ), 
    a)