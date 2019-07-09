from helper import *
import matplotlib.pyplot as plt

def parameters_initialize(layers_size):
    parameters = {}
    L = len(layers_size)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_size[l], layers_size[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layers_size[l], 1))

    return parameters


def forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def forward_activation(A_prev, W, b, activation="relu"):
    if activation == "sigmoid":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activaiton_cache = sigmoid(Z)
    else:
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activaiton_cache = relu(Z)

    cache = (linear_cache, activaiton_cache)
    return A, cache


def forward_probagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = forward_activation(A, parameters["W"+str(l)], parameters['b'+str(l)])
        caches.append(cache)

    A_final, cache = forward_activation(A, parameters["W"+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)


def calculate_cost(A_final, Y):
    m = Y.shape[1]

    cost = (-1 / m) * np.sum((Y * np.log(A_final)) + (1-Y * np.log(1-A_final)))
    cost = np.squeeze(cost)

    return cost


def backward_linear(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.dot(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_activation(dA, cache, activation = "relu"):
    linear_cache, activaiton_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activaiton_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    else:
        dZ = relu_backward(dA, activaiton_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return dA_prev, dW, db

def backwars_probagation(A_final, Y, caches):
    grads = {}
    L = len(caches)
    m = A_final.shape[1]
    Y = Y.reshape(A_final)

    dA_final = - (np.divide(Y, A_final) - np.divide(1 - Y, 1 - A_final))

    current_cache = caches[-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)],  grads["db"+str(L)] = backward_activation(dA_final, current_cache, activation="sigmoid")

    for l in raversed(range(L-1)):
        current_cache = caches[l]
        grads["dA"+str(l)], grads["dW"+str(l+)],  grads["db"+str(l+1)] = backward_activation(dA_final, current_cache)

    return grads

def parameters_update(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1,L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters



def model(X, Y, layers_size, learning_rate = 0.0075, iterations = 3000, print_cost = False):
    costs = []

    parameters = parameters_initialize(layers_size)

    for i in range(iterations):
        A_final, caches = forward_probagation(X, parameters)
        cost = calculate_cost(A_final, Y)

        grads = backwars_probagation(A_final, Y, caches)
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
    m = X.shape[1]
    L = len(parameters) // 2
    p = np.zeros((1,m))

    preds, caches = forward_probagation(X, parameters)

    for i in range(preds.shape[1]):
        if preds[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "+ str(np.sum((p==y)/m)))
    return p
