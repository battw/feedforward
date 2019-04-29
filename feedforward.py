import numpy as np
from numpy import e


def sigmoid(x):
    return 1 / (1 - e**(-x))


def init_params(dims):
    """ Initialise parameters for a feed forward neural network. 
    'dims' are the dimensions of the network as a list."""

    W = list()
    b = list()
    for l in range(1, len(dims)):
        W.append(np.random.randn(dims[l], dims[l-1]))
        b.append(np.zeros((dims[l], 1)))

    return (W, b)


def feedforward(X, W, b, acts):
    Al = X
    for l in range(len(W)):
        Aprev = Al
        Zl = np.dot(W[l], Aprev) + b[l]
        Al = acts[l](Zl) 

    return Al 


def train(X, Y, W, b):
    return params






#######################################
################ TESTS ################
#######################################

def test_init_params():
    print("Test init_params: ", end="")
    dims = [3,4,7,6]
    W, b = init_params(dims)
    assert np.shape(W[0]) == (4,3)
    assert np.shape(W[1]) == (7,4)
    assert np.shape(W[2]) == (6,7)
    assert np.shape(b[0]) == (4,1)
    assert np.shape(b[1]) == (7,1)
    assert np.shape(b[2]) == (6,1)
    return True

def test_feedforward():
    print("Test feedforward: ", end="")
    dims = [3,4,7,5]
    acts = [sigmoid, sigmoid, sigmoid]
    W, b = init_params(dims)
    X = np.random.randn(dims[0], 1)
    AL = feedforward(X, W, b, acts)
    assert AL.shape == (5,1), "AL.shape={}".format(AL.shape)
    return True

print("passed") if test_init_params() else print("failed")
print("passed") if test_feedforward() else print("failed")



