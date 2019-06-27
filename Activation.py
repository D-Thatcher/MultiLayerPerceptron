import numpy as np

def softmax(Z):
    # Z is transposed, thus we perform operations along axis=0
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exps/np.sum(exps, axis=0, keepdims=True)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def leaky_relu(Z):
    return np.maximum(0.01*Z, Z)

def elu(Z):
    return np.where(Z < 0, 1. * (np.exp(Z) - 1), Z)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.
    return dZ

def leaky_relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z < 0] = dZ[Z < 0] * 0.01
    return dZ

def elu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z < 0] = dZ[Z < 0] * elu(Z[Z < 0])
    return dZ


def cross_entropy(Y_hat, Y):
    # Recall that our Y and Y_hat are transposed.
    n_samples = Y .shape[1]
    res = Y_hat - Y
    return res/n_samples


activatation_map = {"softmax":softmax, "sigmoid":sigmoid, "relu":relu, "leaky_relu":leaky_relu, "elu":elu}
backprop_activatation_map = {"sigmoid":sigmoid_backward, "relu":relu_backward, "leaky_relu":leaky_relu_backward, "elu":elu_backward}