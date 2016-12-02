import numpy
import numpy as np
import theano
import pandas as pd
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from load_data import load
from sklearn import preprocessing

def load_data_for_decoder():
    X, y, X_submission, Id = load()
    INPUT_DIM = X.shape[1]    
    X = X.values.astype(numpy.float32)
    y = y.values.astype(numpy.float32)
    X_submission = X_submission.values.astype(numpy.float32)
    y = y.reshape(-1,1)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    X_submission = preprocessing.MinMaxScaler().fit_transform(X_submission)
    return X, X_submission



def load_nn():
    X, y, X_submission, Id = load()
    INPUT_DIM = X.shape[1]    
    X = X.values.astype(numpy.float32)
    y = y.values.astype(numpy.float32)
    X_submission = X_submission.values.astype(numpy.float32)
    y = y.reshape(-1,1)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    X_submission = preprocessing.MinMaxScaler().fit_transform(X_submission)

    X = X.reshape(-1,1,INPUT_DIM) # for mlp and cnn
    X_submission = X_submission.reshape(-1,1,INPUT_DIM) # for mlp and cnn
    return X, y, X_submission, Id




def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


# not working
# net2 = NeuralNet(
    # layers=[  # three layers: one hidden layer
        # ('input', layers.InputLayer),
        # ('conv1', layers.Conv1DLayer),
        # ('pool1', layers.MaxPool1DLayer),
        # ('dropout1', layers.DropoutLayer),
        # ('hidden', layers.DenseLayer),
        # ('output', layers.DenseLayer),
        # ],
    
    
    # # layer parameters:
    # input_shape=(None,1, 410),  # 410 input features per batch
    # conv1_num_filters=10, conv1_filter_size=3,
    # pool1_pool_size=2,
    # dropout1_p=0.1,  # !

    # hidden_num_units=500,  # number of units in hidden layer
    # output_nonlinearity=None,  # output layer uses identity function
    # output_num_units=1,  # 30 target values

    # update_learning_rate=theano.shared(float32(0.01)),
    # update_momentum=theano.shared(float32(0.9)),

    # # optimization method:
    # on_epoch_finished=[
        # AdjustVariable('update_learning_rate', start=0.003, stop=0.0001),
        # AdjustVariable('update_momentum', start=0.9, stop=0.999),
        # EarlyStopping(patience=100),
        # ],

    # regression=True,  # flag to indicate we're dealing with regression problem
    # max_epochs=700,  # we want to train this many epochs
    # verbose=1,
    # )



if __name__ == '__main__':
    np.random.seed(42)

    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        
        # layer parameters:
        input_shape=(None,1, 410),  # 410 input features per batch
        hidden_num_units=200,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=1,  # 1 target values

        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),

        # optimization method:
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.007, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=100),
            ],

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=400,  # we want to train this many epochs
        verbose=1,
        )

    X, y, X_submission, Id = load_nn()
    net1.fit(X, y)
    
    """
    for ipython

    """
    # import matplotlib
    # import matplotlib.pyplot as plt
    # %config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook
    # %matplotlib inline

    # train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    # valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    # plt.plot(train_loss, linewidth=3, label="train")
    # plt.plot(valid_loss, linewidth=3, label="valid")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.ylim(1e-3, 1)
    # plt.yscale("log")
    # plt.show()


    preds = net1.predict(X_submission) 
    y_submission = numpy.expm1(preds).flatten()
    print y_submission.shape
    print y_submission

# computing training error
    from sklearn.metrics import mean_squared_error
    train_preds = net1.predict(X)
    print mean_squared_error(train_preds, y)


    # print "Saving Results."
    # solution = pd.DataFrame({"id":Id, "SalePrice":y_submission})
    # solution.to_csv("nn.csv", index= False)


