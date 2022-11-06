import pytest
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model

input_shape = (28, 28, 1)
epochs = 1
num_classes = 10

@pytest.fixture()
def dense_model(test_data_flattened) -> Sequential:
    model = Sequential()
    model.add(layers.InputLayer(
        input_shape=(input_shape[0] * input_shape[1] * input_shape[2],), name="input_layer"))
    model.add(layers.Dense(350))
    model.add(layers.Dense(num_classes))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    (X_train, Y_train), _ = test_data_flattened
    model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)

    return model


@pytest.fixture()
def generate_conv_model(test_data):
    def _get_conv_model(filters, kernel, use_bias, strides, padding, 
                kernel_initializer='glorot_uniform', activation=None) -> Sequential:
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
        model.add(layers.Conv2D(filters, kernel, use_bias=use_bias, strides=strides, 
                        padding=padding, activation=activation, kernel_initializer=kernel_initializer))
        model.add(layers.Conv2D(filters, kernel, use_bias=use_bias, strides=strides, 
                        padding=padding, activation=activation, kernel_initializer=kernel_initializer))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)

        return model 
    return _get_conv_model


@pytest.fixture()
def generate_depthconv_model(test_data):
    def _generate_depthconv_model(kernel, depth_multiplier, use_bias, strides, padding) -> Sequential:
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
        model.add(layers.DepthwiseConv2D(kernel, 
            depth_multiplier=depth_multiplier, use_bias=use_bias, strides=strides, padding=padding))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)

        return model 
    return _generate_depthconv_model


@pytest.fixture()
def generate_bn_model(test_data):
    def _get_bn_model(axis=3, epsilon=0.001) -> Sequential:
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
        model.add(layers.Conv2D(5, (1, 1)))
        model.add(layers.BatchNormalization(axis=axis, epsilon=epsilon))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _get_bn_model


@pytest.fixture()
def generate_pooling_model(test_data):
    def _get_pooling_model(pool_size, strides, padding) -> Sequential:
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
        model.add(layers.Conv2D(5, (1, 1)))
        model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _get_pooling_model


@pytest.fixture()
def global_pooling_model(test_data) -> Sequential:
    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
    model.add(layers.Conv2D(5, (1, 1)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    (X_train, Y_train), _ = test_data
    model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
    return model


@pytest.fixture()
def generate_zeropadding_model(test_data):
    def _get_zeropadding_model(padding) -> Sequential:
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
        model.add(layers.Conv2D(5, (1, 1)))
        model.add(layers.ZeroPadding2D(padding=padding))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _get_zeropadding_model


@pytest.fixture()
def generate_add_model(test_data):
    def _generate_add_model() -> Sequential:
        input1 = layers.Input(shape=input_shape, name="input_layer")
        x1 = layers.Conv2D(5, (3, 3), strides=2, use_bias=False)(input1)
        #x1 = layers.BatchNormalization()(x1)

        x2 = layers.Conv2D(5, (3, 3), strides=2, use_bias=False)(input1)
        #x2 = layers.BatchNormalization()(x2)

        added = layers.Add()([x1, x2])

        x1 = layers.Conv2D(5, (1, 1), use_bias=False)(added)
        #x1 = layers.BatchNormalization()(x1)

        x2 = layers.Conv2D(5, (1, 1), use_bias=False)(added)
        #x2 = layers.BatchNormalization()(x2)

        added = layers.Add()([x1, x2])
        added = layers.Flatten()(added)

        out = layers.Dense(num_classes)(added)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _generate_add_model


@pytest.fixture()
def generate_conc_model(test_data):
    def _generate_conc_model() -> Sequential:
        input1 = layers.Input(shape=input_shape, name="input_layer")
        x1 = layers.Conv2D(5, (3, 3), strides=2)(input1)
        x1 = layers.BatchNormalization()(x1)

        x2 = layers.Conv2D(5, (3, 3), strides=2)(input1)
        x2 = layers.BatchNormalization()(x2)

        added = layers.Concatenate()([x1, x2])

        x1 = layers.Conv2D(5, (3, 3), padding="same")(added)
        x1 = layers.BatchNormalization()(x1)

        x2 = layers.Conv2D(5, (3, 3), padding="same")(added)
        x2 = layers.BatchNormalization()(x2)

        added = layers.Concatenate()([x1, x2])
        added = layers.Flatten()(added)

        out = layers.Dense(num_classes)(added)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _generate_conc_model


@pytest.fixture()
def generate_resblock_model(test_data):
    def _generate_resblock_model() -> Sequential:
        input1 = layers.Input(shape=input_shape, name="input_layer")
        X = input1
        
        X = layers.Conv2D(32, (7, 7), strides=(2, 2))(X)
        X = layers.BatchNormalization(axis=3)(X)
        #X = layers.Activation('relu')(X)
        X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

        X_shortcut = X

        X = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='valid')(X)
        X = layers.BatchNormalization(axis = 3)(X)
        #X = layers.Activation('relu')(X)

        X = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same')(X)
        X = layers.BatchNormalization(axis = 3)(X)
        #X = layers.Activation('relu')(X)

        X = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='valid')(X)
        X = layers.BatchNormalization(axis = 3)(X)

        X = layers.Add()([X, X_shortcut])
        #X = layers.Activation('relu')(X)
        added = layers.Flatten()(X)

        out = layers.Dense(num_classes)(added)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _generate_resblock_model


@pytest.fixture()
def generate_convblock_model(test_data):
    def _generate_convblock_model() -> Sequential:
        input1 = layers.Input(shape=input_shape, name="input_layer")
        X = input1
        
        X = layers.Conv2D(32, (7, 7), strides=(2, 2))(X)
        X = layers.BatchNormalization(axis=3)(X)
        #X = layers.Activation('relu')(X)
        #X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

        X_shortcut = X

        X = layers.Conv2D(32, (1, 1), strides = (2,2))(X)
        X = layers.BatchNormalization(axis = 3)(X)
        #X = Activation('relu')(X)

        X = layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
        X = layers.BatchNormalization(axis = 3)(X)
        #X = Activation('relu')(X)

        X = layers.Conv2D(filters = 32, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
        X = layers.BatchNormalization(axis = 3)(X)

        X_shortcut = layers.Conv2D(filters = 32, kernel_size = (1, 1), strides = (2,2), padding = 'valid')(X_shortcut)
        X_shortcut = layers.BatchNormalization(axis = 3)(X_shortcut)

        X = layers.Add()([X, X_shortcut])
        #X = layers.Activation('relu')(X)
        added = layers.Flatten()(X)

        out = layers.Dense(num_classes)(added)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), _ = test_data
        model.fit(X_train, Y_train, epochs=epochs, batch_size=250, verbose=0, validation_split=0.2)
        return model
    return _generate_convblock_model


@pytest.fixture(scope="module", autouse=True)
def test_data():
    print("test_data generated")
    (X_train, Y_train), (X_val, Y_val) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1).astype('float32') / 255
    X_val = np.expand_dims(X_val, axis=-1).astype('float32') / 255
    Y_train = to_categorical(Y_train, num_classes)
    Y_val = to_categorical(Y_val, num_classes)

    return (X_train, Y_train), (X_val, Y_val)


@pytest.fixture(scope="module", autouse=True)
def test_data_flattened():
    print("test_data_flattened generated")
    (X_train, Y_train), (X_val, Y_val) = mnist.load_data()

    X_train = np.reshape(X_train, (X_train.shape[0],-1)).astype('float32') / 255
    X_val = np.reshape(X_val, (X_val.shape[0],-1)).astype('float32') / 255
    Y_train = to_categorical(Y_train, num_classes)
    Y_val = to_categorical(Y_val, num_classes)
    
    return (X_train, Y_train), (X_val, Y_val)
