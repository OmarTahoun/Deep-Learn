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
