import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from lasagne.nonlinearities import sigmoid
from nn import AdjustVariable
from nn import EarlyStopping
ENCODE_FEATURE_NAME= 'encode'

def float32(k):
        return np.cast['float32'](k)


def fit(X):
    INPUT_DIM = X.shape[1]
    # X = X.reshape(-1, INPUT_DIM)

    encoder = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('encode', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        
        # layer parameters:
        input_shape=(None, INPUT_DIM),  # 410 input features per batch
        encode_num_units=100,  # number of units in hidden layer
        output_nonlinearity=sigmoid,  # output layer uses identity function
        output_num_units=INPUT_DIM,  # 1 target values

        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),

        # optimization method:
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.007, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=100),
            ],

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=300,  # we want to train this many epochs
        verbose=1,
        )

    encoder.fit(X, X)
    return encoder


def fit_transform(X):
    encoder = fit(X)
    cleaned_X = encoder.predict(X)
    return cleaned_X

def get_layer_by_name(net, name):
    for i, layer in enumerate(net.get_all_layers()):
        if layer.name == name:
            return layer, i
    return None, None

def encode_input(encode_layer, X):
    return layers.get_output(encode_layer, inputs=X).eval()

def get_encoded_feature(X):
    encoder = fit(X)
    encode_layer, encode_layer_index = get_layer_by_name(encoder, 'encode')
    # print encode_layer, encode_layer_index
    X_encoded = encode_input(encode_layer, X)
    # print X_encoded.shape
    return X_encoded






