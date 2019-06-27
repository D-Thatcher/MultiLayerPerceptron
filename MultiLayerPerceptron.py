import numpy as np
import os
import matplotlib.pyplot as plt
from Optimizer import AdamOptimizer, GradientDescent
from Activation import activatation_map, backprop_activatation_map, cross_entropy
from timeit import default_timer as timer


def get_accuracy_value(Y_hat, Y):
    return (Y_hat.argmax(axis=0) == Y.argmax(axis=0)).mean()


def split_train_test(data, target, seed=23, test_size= 0.1):
    np.random.seed(seed=seed)
    h,w = data.shape
    th, tw = target.shape

    assert h == th, "Unequal number of samples amd number of target class"
    assert float(target.max()) == 1, "target ndarray must be in one-hot format"

    num_test = int(test_size*h)

    indices = np.arange(h)
    np.random.shuffle(indices)

    test_ind = indices[:num_test]
    train_ind = indices[num_test:]

    X_train, X_test, y_train, y_test = data[train_ind,:], data[test_ind,:], target[train_ind,:], target[test_ind,:]

    return X_train, X_test, y_train, y_test


def build_one_hot(enum_class_array):
    enum_class_array = np.array(enum_class_array).astype(np.int64).flatten()

    if 0 not in enum_class_array:
        enum_class_array -= 1

    ret = np.zeros((len(enum_class_array),enum_class_array.max()+1)).astype(np.float64)
    ret[list(range(len(enum_class_array))), list(enum_class_array)] = 1.0
    return ret


def error(pred, real):
    n_samples = real.shape[1]
    logp = - np.log(pred[real.argmax(axis=0), np.arange(n_samples)])
    loss = np.sum(logp) / n_samples
    return loss


def map_to_unit(nd):
    return (nd -nd.min())/(nd.max() - nd.min())


class MLP:
    def __init__(self, optimizer='AdamOptimizer', learning_rate=0.001, config=None, seed=10):
        self.configuration = []
        self.cost_history = []
        self.accuracy_history = []
        self.learning_rate = learning_rate
        self.optimizer = self.load_optimizer(optimizer)
        if not (config is None):
            self.set_layers(config)
        # self.dropout_p = dropout_p
        self.seed = seed
        self.params = None

        # Metadata for saving the model
        self.save_enabled = False
        self.save_path = ""
        self.epoch_per_save = -1

    def _save_state(self):
        assert self.save_enabled, "Enable saving before training using the enable_save method"
        state = {"configuration":self.configuration, "params":self.params, "epoch_per_save":self.epoch_per_save,"opt_state": self.optimizer.pull_state()}
        np.save(self.save_path, state)

    def enable_save(self, save_path="", epoch_per_save=500):
        assert epoch_per_save>0, "epoch_per_save must be a positive integer"
        self.save_path = save_path
        self.save_enabled = True
        self.epoch_per_save = epoch_per_save

    def disable_save(self):
        self.save_enabled = False

    # Implicitly enables save state
    def load_model(self, load_path):
        assert load_path[-4:] != "npy", "Unexpected file format. Expected file of type '.npy'"
        assert os.path.exists(load_path), "File not found"

        state = np.load(load_path)

        # Access dict of numpy array using [()] operator prefix
        self.configuration = state[()]['configuration']
        self.params = state[()]['params']
        self.epoch_per_save = state[()]['epoch_per_save']
        self.optimizer.push_state(state[()]['opt_state'])

        self.save_path = load_path
        self.save_enabled = True


    def add_layer(self, input_n, output_n, activation):
        assert input_n>0
        assert output_n>0
        assert callable(activation)
        self.configuration.append({"in": input_n, "out": output_n, "func": activation})

    @staticmethod
    def make_explicit_config(config):
        assert len(config) > 1
        explicit_config = []

        for idx in range(0, len(config) - 1):
            explicit_config.append(
                {"in": config[idx]["neuron"], "out": config[idx + 1]["neuron"], "func": config[idx]["activation"]})
        return explicit_config

    def set_layers(self,layers):
        if "in" in layers:
            self.configuration = layers
        else:
            self.configuration = self.make_explicit_config(layers)

    def load_optimizer(self, name):
        if name is 'AdamOptimizer':
            return AdamOptimizer(learning_rate=self.learning_rate)
        elif name is 'GradientDescent':
            return GradientDescent(learning_rate=self.learning_rate)
        else:
            raise Exception('Optimizer must be one of AdamOptimizer or GradientDescent')


    def init_all_layers(self):
        np.random.seed(self.seed)

        self.params = {}

        # For each layer, initialize the matrix using Xavier Initialization
        for idx, layer in enumerate(self.configuration):

            # Layers are enumerated from 1 to num_layers
            layer_idx = idx + 1

            # This layer's input neurons
            layer_input_size = layer["in"]

            # This layer's output neurons
            layer_output_size = layer["out"]


            # Xavier Uniform initialization of matricies
            self.params['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * np.sqrt(2. / (layer_output_size + layer_input_size))
            self.params['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * np.sqrt(2. / (layer_output_size + 1))

        return

    def layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        # Essentially, forward propagation has two responsibilities:
        #           (1) Calculating Z = W*A + b
        #               Where, A is the previous layer's activated matrix and Z will be this layer's activated matrix
        #
        #           (2) Returning the activation function of Z and Z itself.

        # The order of our multiplication is one of the reasons for the necessary transposing of
        # our initially inputted X and Y in the 'train' method
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # TODO Apply dropout
        # if not (self.dropout_p is None) and self.training:
        #     u1 = np.random.binomial(1, self.dropout_p, size=Z_curr.shape) / self.dropout_p
        #     Z_curr *= u1

        # Get the activation function pointer
        assert activation in activatation_map, 'Unknown activation function'
        activation_func = activatation_map[activation]

        # Return activated Z matrix and Z matrix
        return activation_func(Z_curr), Z_curr

    def multilayer_forward_propagation(self, X):

        # Cache our intermediary matrices to allow for their use in backpropagation
        memory = {}

        # Initialize the first layer's pseudo activated matrix
        A_curr = X

        # For each layer, perform the single layer propagation method and store the results
        for idx, layer in enumerate(self.configuration):

            layer_idx = idx + 1

            # Store the previous activation
            A_prev = A_curr

            # Get activation function pointer for this layer
            activ_function_curr = layer["func"]

            # Get W for this layer
            W_curr = self.params["W" + str(layer_idx)]

            # Get b for this layer
            b_curr = self.params["b" + str(layer_idx)]

            # Multiply and activate the matrix
            A_curr, Z_curr = self.layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            # Cache the matrices
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector and a dictionary containing intermediate values
        return A_curr, memory



    def layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):

        # number of samples
        m = float(A_prev.shape[1])

        # Get the backprop activation function pointer
        if not (activation == "softmax"):

            assert activation in activatation_map, 'Unknown activation function'
            backward_activation_func = backprop_activatation_map[activation]

            # Activation function derivative
            dZ_curr = backward_activation_func(dA_curr, Z_curr)
        else:
            # Softmax does not need to be differentiated
            dZ_curr = dA_curr

        # Differentiate each of the matrices and return them
        dW_curr = np.dot(dZ_curr, A_prev.T) / m

        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m

        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr




    def multilayer_backward_propagation(self, Y_hat, Y, memory):
        grads_values = {}

        dA_prev = cross_entropy(Y_hat, Y)
        # dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));

        for layer_idx_prev, layer in reversed(list(enumerate(self.configuration))):

            layer_idx_curr = layer_idx_prev + 1

            # Get the key for the backprop activation function pointer for this layer
            activ_function_curr = layer["func"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = self.params["W" + str(layer_idx_curr)]
            b_curr = self.params["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["W" + str(layer_idx_curr)] = dW_curr
            grads_values["b" + str(layer_idx_curr)] = db_curr

        return grads_values

    def train(self, X, Y, epochs, log=False):
        if log: start = timer()
        assert len(self.configuration) > 1

        # Our helper methods and multiplication orders require the transposition of our data and target
        X = X.T
        Y = Y.T

        # Initiation of each layers' matrices if the model wasn't loaded or trained once before
        if self.params is None:
            self.init_all_layers()

        # For each epoch, run the forward propagation -> backward propagation -> gradient update cycle
        for i in range(epochs):
            # step forward
            Y_hat, cache = self.multilayer_forward_propagation(X)

            # Store measures
            cost = error(Y_hat, Y)
            self.cost_history.append(cost)

            accuracy = get_accuracy_value(Y_hat, Y)
            self.accuracy_history.append(accuracy)

            # Perform backpropagation
            grads_values = self.multilayer_backward_propagation(Y_hat, Y, cache)

            # Call the optimizer to update the params using the grads
            self.params = self.optimizer(self.params, grads_values)

            if (i % 50 == 0) and log:
                print('Epoch  ', i, '  Cost  ', cost, '  Accuracy  ', accuracy)

            if self.save_enabled and (i % self.epoch_per_save == 0) and (i != 0):
                if log:
                    print('Saving Model...')
                self._save_state()

        if log:
            print("MLP trained in " + str(timer() - start) + " seconds")

        return

    def cost_visual(self):
        plt.plot(self.cost_history)
        plt.title('Entropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()

    def accuracy_visual(self):
        plt.plot(self.accuracy_history)
        plt.title('Classification Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Fraction of Correct Predictions on Training Data')
        plt.show()
