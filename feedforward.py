""" """
import numpy as np
from numpy import e


def sigmoid(x):
    return 1 / (1 + e**(-x))

def dsigmoid(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def sum_sqr_err(AL, Y):
    return np.sum((AL - Y)**2, axis=1, keepdims=True) / (2 * len(Y))

def dsum_sqr_err(AL, Y):
    return np.sum(AL - Y, axis=1, keepdims=True) / len(Y) 

def dsqr_loss(AL, Y):
    return AL - Y

def init(n):
    """ Initialise parameters for a feed forward neural network. 
    'n' is a list containing the dimension of each layer."""
    L = len(n) # number of layers
    W = [None] * L 
    b = [None] * L 
    for l in range(1, L):
        W[l] = np.random.randn(n[l], n[l-1])
        b[l] = np.zeros((n[l], 1))

    return (W, b)


def feedforward(X, W, b, acts):
    L = len(W) # number of layers
    acts = [None] + acts
    A = [X] + (L - 1)*[None]
    Z = L * [None]  
    for l in range(1, len(W)):
        Z[l] = np.dot(W[l], A[l-1]) + b[l]
        A[l] = acts[l](Z[l])

    return A, Z 


def backprop(W, Y, A, Z, d_acts, d_loss):
    L = len(W) # number of layers
    m = Y.shape[1]

    d_acts = [None] + d_acts
    dA = (L-1) * [None] + [d_loss(A[-1], Y)] 
    dZ = L * [None] 

    dW = L * [None]
    db = L * [None]

    for l in range(L - 1, 0, -1):
        dZ[l] = dA[l] * d_acts[l](Z[l])
        dA[l-1] = np.dot(W[l].T, dZ[l])

        dW[l] = np.dot(dZ[l], A[l-1].T) / m
        db[l] = np.sum(dZ[l], axis=1, keepdims=True) / m

    return dW, db
                

def update(W, b, dW, db, lrate):
    L = len(W)
    prev_W = W
    prev_b  = b
    W = L * [None] 
    b = L * [None]
    for l in range(1, len(W)):
        W[l] = prev_W[l] - lrate * dW[l]
        b[l] = prev_b[l] - lrate * db[l]
        
    return W, b


def train(X, Y, W, b, acts, d_acts, loss, d_loss, lrate, iters):
    A, Z  = feedforward(X, W, b, acts)
    dW, db = backprop(W, Y, A, Z, d_acts, d_loss)
    W, b = update(W, b, dW, db, lrate)
    return W, b 


def batch_train(batch_size, X, Y, W, b, acts, d_acts, loss, d_loss, lrate):
    for i in range(X.shape[1] // batch_size):
        start = i * batch_size
        end = min(batch_start + batch_size, X.shape[1](X) - 1)
        W, b = train_batch(X[:, start:end], Y[start:end], W, b, acts, lrate)

    return W, b




#######################################
################ TESTS ################
#######################################

# These tests check that the resulting matrices have the correct dimensions.
def test_init():
    print("Test init: ", end="")
    dims = [3,4,7,6]
    W, b = init(dims)
    assert np.shape(W[1]) == (4,3), np.shape(W[1])
    assert np.shape(W[2]) == (7,4), np.shape(W[2])
    assert np.shape(W[3]) == (6,7), np.shape(W[3])
    assert np.shape(b[1]) == (4,1), np.shape(b[1])
    assert np.shape(b[2]) == (7,1), np.shape(b[2])
    assert np.shape(b[3]) == (6,1), np.shape(b[3])
    return True

def test_feedforward():
    print("Test feedforward: ", end="")
    m = 10 
    dims = [3,4,7,5]
    acts = [sigmoid, sigmoid, sigmoid]
    W, b = init(dims)
    X = np.random.randn(dims[0], m)
    A, Z = feedforward(X, W, b, acts)
    assert A[0].shape == (3,m), A[0].shape 
    assert A[1].shape == (4,m), A[1].shape 
    assert A[2].shape == (7,m), A[2].shape 
    assert A[3].shape == (5,m), A[3].shape 

    assert Z[1].shape == (4,m), Z[1].shape 
    assert Z[2].shape == (7,m), Z[2].shape 
    assert Z[3].shape == (5,m), Z[3].shape 

    return True

def test_backprop():
    print("Test backprop: ", end="")
    m = 19
    dims = [3,4,7,5]
    acts = [sigmoid, sigmoid, sigmoid]
    W, b = init(dims)
    X = np.random.randn(dims[0], m)
    Y = np.ones((dims[-1], m))
    A, Z = feedforward(X, W, b, acts)
    dW, db = backprop(W, Y, A, Z, [dsigmoid, dsigmoid, dsigmoid], dsqr_loss)
    assert len(dW) == 4,  len(dW)
    assert dW[1].shape == (4, 3), dW[1].shape
    assert dW[2].shape == (7, 4), dW[2].shape
    assert dW[3].shape == (5, 7), dW[3].shape

    assert len(db) == 4,  len(db)
    assert db[1].shape == (4, 1), db[1].shape
    assert db[2].shape == (7, 1), db[2].shape
    assert db[3].shape == (5, 1), db[3].shape
    return True

def test_update():
    print("Test update: ", end="")
    m = 19
    dims = [3,4,7,5]
    acts = [sigmoid, sigmoid, sigmoid]
    W, b = init(dims)
    X = np.random.randn(dims[0], m)
    Y = np.ones((dims[-1], m))
    A, Z = feedforward(X, W, b, acts)
    dA, dZ = backprop(W, Y, A, Z, [dsigmoid, dsigmoid, dsigmoid], dsqr_loss)
    W, b = update(W, b, dA, dZ, 0.001)
    assert W[1].shape == (4,3), W[1].shape
    assert W[2].shape == (7,4), W[2].shape
    assert W[3].shape == (5,7), W[3].shape

    assert b[1].shape == (4,1), b[1].shape
    assert b[2].shape == (7,1), b[2].shape
    assert b[3].shape == (5,1), b[3].shape
    return True


def run_test(test_func):
        print("passed") if test_func() else print("failed")

run_test(test_init) 
run_test(test_feedforward)
run_test(test_backprop)
run_test(test_update)
        




