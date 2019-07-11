import numpy as np

def sigmoid(Z):
    """
    Implementation of the sigmoid function in numpy

    Args: Z -- a Numpy array (a vector a Matrix. Any shape is okay...no discrimination)

    Return:
        A -- The output of the sigmoid function (Same shape as Z)
        Z -- Z is returned, it's used later during backpropagation.
    """

    A = 1 / (1+np.exp(-Z))
    return A, Z


def relu(Z):
    """
    Implementation of the relu function in numpy

    Args: Z -- a Numpy array (a vector a Matrix. Any shape is okay...no discrimination)

    Return:
        A -- The output of the relu function (Same shape as Z)
        Z -- Z is returned, it's used later during backpropagation.
    """

    A = np.maximum(0, Z)
    return A, Z


def relu_backward(dA, cache):
    """
    Implementation the backward backward probagation of the relu function in numpy

    Args:
            dA      -- The gradient of an output fron a layer
            cache   -- 'Z' we stored , computing backward probabagation is more efficient

    Return: dZ      -- Cost gradient in respect to 'Z'
    """

    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implementation the backward backward probagation of the sigmoid function in numpy

    Args:
            dA      -- The gradient of an output fron a layer
            cache   -- 'Z' we stored , computing backward probabagation is more efficient

    Return: dZ      -- Cost gradient in respect to 'Z'
    """

    s = 1/(1+np.exp(-cache))
    dZ = dA * s * (1-s)

    return dZ


def predict(X, y, parameters):
    """
    Predicts the result of L-layer deep neural network

    Args:
            X           -- dataset samples to be labeled
            y           -- True labels of the samples
            parameters  -- parameters of a trained model

    Return: p           -- predictions for the dataset X
    """

    m = X.shape[1]
    L = len(parameters) // 2 # The number of layers in the network
    p = np.zeros((1,m))

    # Forward probagation
    preds, caches = forward_probagation(X, parameters)

    # converting the probabilitis to 0/1 predictions
    for i in range(preds.shape[1]):
        if preds[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # Print the Accuracy of this model on the given dataset
    print("Accuracy: "+ str(np.sum((p==y)/m)))
    return p
