import numpy as np

def sigmoid(z):
    A = 1 / (1+np.exp(-z))
    return A, z

def relu(z):
    A = np.maximum(o, z)
    return A, z

def relu_backward(dA, cache):
    dZ = np.array(dA, copy = True)
    dZ[cache <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):
    s = sigmoid(cache)
    dZ = dA * s * (1-s)

    return dZ
