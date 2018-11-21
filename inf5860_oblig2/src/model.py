#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 2 in                                             #
# INF5860 - Machine Learning for Image analysis                                 #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2018.03.01                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde 


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1
    l_dims = conf["layer_dimensions"]
    
#    W = {"W_" + str(i+1): np.random.normal(loc=0, scale=np.sqrt(2/xy[0]), size=(xy[0], xy[1])) 
#            for i, xy in enumerate(zip(l_dims[:-1], l_dims[1:]))}
#    
#    b = {"b_" + str(i+1): np.zeros((x, 1)) for i, x in enumerate(l_dims[1:])}
#
#    params = {**W, **b}
    
    params = {}

    for i in range(len(l_dims)-1):
        params["W_" + str(i+1)] = np.random.normal(loc=0.0, scale=np.sqrt(2/l_dims[i]), 
              size=(l_dims[i], l_dims[i+1]))
        params["b_" + str(i+1)] = np.zeros((l_dims[i+1], 1))

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 a)
    if activation_function == 'relu':
        return np.maximum(0, Z)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 b)
    return np.exp(Z - np.amax(Z)) / np.sum(np.exp(Z - np.amax(Z)), axis=0)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 2 c)
    Y_proposed = None
    features = {}
    
    L = len(conf["layer_dimensions"])
    activation_function = conf["activation_function"]
    
    activations = X_batch
    features["A_0"] = X_batch

    for i in range(1, L):
        Z = np.dot(params["W_" + str(i)].T, activations) + params["b_" + str(i)]
        features["Z_" + str(i)] = Z


        if i == (L - 1):
            A = softmax(Z)
            Y_proposed = A
        else:
            A = activation(Z, activation_function)
            features["A_" + str(i)] = A
            activations = A

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 3
    n_y, m = Y_proposed.shape
    
    cost = (-1/m) * np.sum(np.sum(Y_reference * np.log(Y_proposed), axis=1))
    num_correct = np.sum(np.argmax(Y_proposed, axis=0) == np.argmax(Y_reference, axis=0))
    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 4 a)
    if activation_function == 'relu':
        return (Z >= 0) * 1
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 4 b)
        
    L = len(conf["layer_dimensions"])
    activ_func = conf["activation_function"]
    m = Y_proposed.shape[1]
    
    grad_params = {}
    cost = Y_proposed - Y_reference
    
    for i in range(L-1, 0, -1):
        
        if i < (L-1):
            cost = activation_derivative(features["Z_" + str(i)], activ_func) * np.matmul(params["W_" + str(i+1)], cost)
        
        grad_b = (1/m) * np.dot(cost, np.ones((m, 1)))
        grad_W = (1/m) * np.matmul(features["A_" + str(i-1)], cost.T)
        
        grad_params["grad_b_" + str(i)] = grad_b
        grad_params["grad_W_" + str(i)] = grad_W

    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        updated_params: Updated parameter dictionary.
    """
    # TODO: Task 5
    L = len(conf["layer_dimensions"])
    lamda = conf["learning_rate"]
    updated_params = {}
    
    for i in range(L-1, 0, -1):
        updated_params["W_" + str(i)] = params["W_" + str(i)] - lamda * grad_params["grad_W_" + str(i)]
        updated_params["b_" + str(i)] = params["b_" + str(i)] - lamda * grad_params["grad_b_" + str(i)]

    return updated_params
