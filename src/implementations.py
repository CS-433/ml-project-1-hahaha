'''File contains all function implementations from table 1 of step 2 '''

# Useful starting lines
import numpy as np
from proj1_helpers import *

"""Functions"""

# Cost functions mse
def compute_loss_mse(y, tx, w):
    """calculate loss using mean squared error"""
    err = y - tx.dot(w)
    return 1/2 * np.mean(err**2)


# Gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -1 / len(err) * tx.T.dot(err)
    return grad, err


# least_squares_GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        
        # update w by gradient descent
        w = w - gamma * grad

    return  w, compute_loss_mse(y, tx, w)

# shuffle data in random
def shuffle_dataset(y, tx):
    """shuffling dataset"""

    # np.random.seed(1) #if commented selects every time you run a different seed
    random_shuffle = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[random_shuffle]
    shuffled_tx = tx[random_shuffle]

    return shuffled_y, shuffled_tx

# least square with SGD
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Computes least squares using Stochastic Gradient Descent"""
    w = initial_w
    shuffled_y, shuffled_tx = shuffle_dataset(y, tx)

    for n_iter in range(max_iters):
        
        # each step contains 1 datapoint
        for training_example in range(len(y)):
            e = shuffled_y[training_example] -shuffled_tx[training_example].dot(w)
            grad = -e * shuffled_tx[training_example]
            w = w - gamma * grad

    return w, compute_loss_mse(shuffled_y, shuffled_tx, w)


# least_squares
def least_squares(y, tx):
    """calculate the least squares."""

    # calculate w
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)

    return w, compute_loss_mse(y, tx, w)

# ridge_regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    # large model w wi will be penalized
    lambda_aux = lambda_ * (2*len(y)) 
    A = np.dot(tx.T, tx) + lambda_aux * np.eye(tx.shape[1])    
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)
   
    return w, compute_loss_mse(y, tx, w)

# loss function for logistic_regression
def compute_loss_logistic_regression(y, tx, w):
    """calculate loss for logistic regression"""
    sigmoid = 1 / (1 + np.exp(-(tx.dot(w))))
    loss = -1 / len(y) * np.sum((1 - y) * np.log(1 - sigmoid) + y * np.log(sigmoid))
    return loss

# logistic_regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Computes logistic regression using gradient descent"""
    w = initial_w

    for n_iter in range(max_iters):
        # apply sigmoid function on tx @ w
        sigmoid = 1/ (1 + np.exp(-(tx.dot(w))))
        gradient = -1/len(y) * tx.T.dot(y-sigmoid)
        w = w-gamma * gradient

    loss = compute_loss_logistic_regression(y, tx, w)
    return w, loss

# regularized logistic regression
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Computes regularized logistic regression using gradient descent"""
    w = initial_w

    for n_iter in range(max_iters):
        sigmoid = 1 / (1 + np.exp(-(tx.dot(w))))
        gradient = -1 / len(y) * tx.T.dot(y - sigmoid) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = compute_loss_logistic_regression(y, tx, w) + lambda_ * w.T.dot(w)
    return w, loss
