from MultiLayerPerceptron import *
import numpy as np
from sklearn.datasets import load_digits

# Load the data
digits = load_digits()

# Build a one-hot class representation from our enumerated class 1-D array using the build_one_hot method
onehot_target = build_one_hot(digits.target)

# Split the data into a test and train group. Test size is the fraction of the data that will be used to
# evaluate the test accuracy. The seed is an integer that makes the results reproducible.
X_train, X_test, y_train, y_test = split_train_test(digits.data, onehot_target, test_size=0.1, seed=13)


y_train = np.array(y_train)
y_test = np.array(y_test)

# Map the data to the unit interval.
X_train = map_to_unit(X_train)
X_test = map_to_unit(X_test)

# Our data is a collection of 8x8 images. Each image (ndarray) is flattened into a 64 dimension array
NUM_PIXELS = 64

# The digits correpsond to the integers 0 through 9
NUM_CLASS = 10

# Either of the following configuration formats are permissible (and in this case, identical)
explicit_configuration = [
    {"in": NUM_PIXELS, "out": 75, "func": "relu"}, # Input layer
    {"in": 75, "out": 60, "func": "relu"},
    {"in": 60, "out": 50, "func": "relu"},
    {"in": 50, "out": 25, "func": "relu"},
    {"in": 25, "out": NUM_CLASS, "func": "softmax"},  # Output layer
]

configuration = [
    {"neuron": NUM_PIXELS, "activation": "relu"}, # Inputted data
    {"neuron": 75, "activation": "relu"},
    {"neuron": 60, "activation": "relu"},
    {"neuron": 50, "activation": "relu"},
    {"neuron": 25, "activation": "softmax"},
    {"neuron": NUM_CLASS, "activation": ""},  # Outputted predictions
]


# Optimizer can be one of 'AdamOptimizer' or 'GradientDescent'
neural_net = MLP(optimizer='AdamOptimizer', learning_rate=0.001, seed=21)

# Enable the model to save itself to save_path every epoch_per_save epochs
neural_net.enable_save(save_path=r"mlp_model.npy",
                       epoch_per_save=50)

# Set the configuration using 'set_layers' or add layers individually using 'add_layer'
# Note that you can set_layers with an implicit configuration (see 'configuration' above),
# or explicit configuration (see 'explicit_configuration' above)
neural_net.set_layers(configuration)

# Begin the training 85% train accuracy in 100 epoch
neural_net.train(X_train, y_train, epochs=100, log=True)

# Evaluate our model on data that hasn't been used to train the model. Notice the difference in accuracy.
Y_test_hat, _ = neural_net.multilayer_forward_propagation(np.transpose(X_test))

# Recall that our data and target arrays are transposed inside the train  method, so Y_test_hat is outputted transposed.
# Let's transpose y_test accordingly and evaluate the accuracy.
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test))

print("Test batch accuracy: " + str(acc_test))

# Plot the cost vs epoch
neural_net.cost_visual()

# Plot the accuracy vs epoch
neural_net.accuracy_visual()

# Reload the neural net
print('Reloaded')
reloaded_neural_net = MLP()
reloaded_neural_net.load_model(load_path=r"mlp_model.npy")

# Saving is implicitly enabled after loading our previous model, so let's disable it
reloaded_neural_net.disable_save()

reloaded_neural_net.train(X_train, y_train, epochs=150, log=True)

Y_test_hat, _ = reloaded_neural_net.multilayer_forward_propagation(np.transpose(X_test))

improved_acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test))

print("Test batch improved accuracy: " + str(improved_acc_test))

# TODO Report FP, TP, TN, TP, Confusion Matrix Visual, F1 Score, option for save report, lightweight predictor
