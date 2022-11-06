import tensorflow as tf

def build_model(model_name, model_weights, input_shape, classes=None, trainable=False):
    params = {
        "input_shape": input_shape, 
        "weights": model_weights
    }
    if classes is not None:
        params["classes"] = classes
    #params["classifier_activation"] = None

    model_name = model_name.lower()
    if "vgg16" in model_name:
        ModelInstance = tf.keras.applications.VGG16
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    elif "vgg19" in model_name:
        ModelInstance = tf.keras.applications.VGG19
        preprocess_input = tf.keras.applications.vgg19.preprocess_input

    elif "resnet50" in model_name:
        ModelInstance = tf.keras.applications.ResNet50
        preprocess_input = tf.keras.applications.resnet.preprocess_input

    elif "densenet" in model_name:
        ModelInstance = tf.keras.applications.densenet.DenseNet201
        preprocess_input = tf.keras.applications.densenet.preprocess_input

    elif "mobilenet" in model_name:
        ModelInstance = tf.keras.applications.mobilenet.MobileNet 
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        
    elif "inception" in model_name:
        ModelInstance = tf.keras.applications.inception_v3.InceptionV3 
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        
    elif "custom" in model_name:
        model = tf.keras.models.load_model(str(model_weights)) 
        preprocess_input = (lambda x: x / 255.0)
        
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
        return model, preprocess_input, (lambda x: x) 
        
    else:
        return

    model = ModelInstance(**params)
    
    #model.trainable = trainable
    #for layer in model.layers:
    #    layer.trainable = trainable
    #model.summary()
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = False
    return model, preprocess_input, tf.keras.applications.imagenet_utils.decode_predictions


