from helper import *
import matplotlib.pyplot as plt

def parameters_initialize(layers_size):
    """
    Initializes the parameters of the neural network (Weights and biases)

    Args:   layers_size -- the sizes of the layers in your neural network, in the form of a python list

    Return: parameters  S-- a dictionary containing the weights and biases of each layer (W1, b1, W2, b2, ...)
    """

    parameters = {}
    L = len(layers_size)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_size[l], layers_size[l-1]) * np.sqrt(np.divide(2,layers_size[l-1]))
        parameters["b"+str(l)] = np.zeros((layers_size[l], 1))

    return parameters


def forward_linear(A, W, b):
    """
    Implementation of the linear part of the forward probagation algorithm.

    Args:
        A -- The activation (output) from the previous layer  (numpy array shape of previous layer, number of samples).
        W -- The weights matrix (numpy array shape of current layer, size of previous layer)
        b -- The bias vector,  (numpy array shape of current layer, 1)

    Return:
        Z     -- the pre-activation parameter, the input to the activation function (Relu, Sigmoid, ...)
        cache -- a dictionary containing (A, W, b) used later in backpropagation
    """

    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache


def forward_activation(A_prev, W, b, activation="relu"):
    """
    Implementation of the activation part of forward probagation.

    Args:
        A_prev      -- The output of the previous layer (pre-activation parameter)
        W           -- The weights matrix (numpy array shape of current layer, size of previous layer)
        b           -- The bias vector,  (numpy array shape of current layer, 1)
        activation  -- The method of activation (default = 'relu')

    Return:
        A       -- The output of the activaiton function (post-activation value)
        cache   -- a dictionary containing the values we used (linear cache, activaiton cache) used later in back propagation.
    """

    if activation == "sigmoid":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activaiton_cache = sigmoid(Z)
    else:
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activaiton_cache = relu(Z)

    cache = (linear_cache, activaiton_cache)
    return A, cache


def forward_probagation(X, parameters):
    """
    Implementation of the forward probagation function from start to end [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    Args:
        X           -- The training dataset, numpy array of shape (input size, numper of samples)
        parameters  -- output of parameters_initialize function()

    Return:
        A_final -- The output of the final layer (last post-activation value in the network)
        caches  -- list of the caches for every layer.
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = forward_activation(A_prev, parameters["W"+str(l)], parameters['b'+str(l)])
        caches.append(cache)

    A_final, cache = forward_activation(A, parameters["W"+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)

    return A_final, caches


def calculate_cost(A_final, Y):
    """
    Implementation of the cost function.

    Args:
        A_final -- probabilitis vector corresponding to predictions of labels, shape(1, numper of samples)
        Y       -- The true label vactor

    Return: cost -- cross-entropy cost
    """

    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y,np.log(A_final).T) - np.dot(1-Y, np.log(1-A_final).T))
    cost = np.squeeze(cost)

    return cost


def backward_linear(dZ, cache):
    """
    Implementation of the linear part of back probagation.

    Args:
        dZ      -- cost gradient in relation to the linear output of the previous layer.
        cache   -- (A_prev, W, b) stored by forward probagation of this layer.

    Return:
        dA_prev -- gradient of the cost in relation to the activation of the previous layer.
        dW      -- gradient of the cost in relation to W of the current layer.
        db      -- gradient of the cost in relation to b of the current layer.
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_activation(dA, cache, activation = "relu"):
    """
    Implementation of the backward probagation activation part.

    Args:
        dA          -- post-activation gradient of the current layer.
        cache       -- (linear cache, activaiton chache) stored by the forward activaiton of this layer.
        activaiton  -- the activation function used (default = 'relu')

    Return:
        dA_prev -- cost gradient in relation to the activation of the previous layer
        dW      -- gradient of the cost in relation to W of the current layer.
        db      -- gradient of the cost in relation to b of the current layer.
    """

    linear_cache, activaiton_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activaiton_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    else:
        dZ = relu_backward(dA, activaiton_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return dA_prev, dW, db


def backward_probagation(A_final, Y, caches):
    """
    Implementation of the backward probagation function.

    Args:
        A_final -- probabilitis vector, from forward probagation.
        Y -- true label vector
        caches -- list of caches.

    Return: grads -- a dictionary containing the gradients of all layers.
    """

    grads = {}
    L = len(caches)
    m = A_final.shape[1]
    Y = Y.reshape(A_final.shape)

    dA_final = - (np.divide(Y, A_final) - np.divide(1 - Y, 1 - A_final))

    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)],  grads["db"+str(L)] = backward_activation(dA_final, current_cache, activation="sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def parameters_update(parameters, grads, learning_rate):
    """
    Updates the parameters after calculating the gradients.

    Args:
        parameters      -- the original parameters python dictionary
        grads           -- the gradients of the parameters, python dictionary
        learning_rate   -- the hyperparameter alpha or the learning rate (default = 0.0075)

    Return: parameters -- the parameters after being updated.
    """
    L = len(parameters) // 2

    for l in range(1,L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

        return parameters



def model(X, Y, layers_size, learning_rate = 0.0075, iterations = 3000, print_cost = False):
    """
    Implemnets a L-layer neural network.

    Args:
        X -- Training dataset input
        Y -- Training dataset labels
        layers_size -- The dimenssions of the layers in the network
        learning_rate -- learning rate of the update rule (optional),  default = 0.0075
        iterations -- the number of training iterations (optional),  default = 3000
        print_cost -- if true the cost will be printed every 100 iteration (optional),  default = false
    """

    costs = []
    parameters = parameters_initialize(layers_size)

    for i in range(0, iterations):
        A_final , caches = forward_probagation(X, parameters)
        cost = calculate_cost(A_final, Y)

        grads = backward_probagation(A_final, Y, caches)
        parameters = parameters_update(parameters, grads, learning_rate)

        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning Rate = " + str(learning_rate))
    plt.show()

    return parameters


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
