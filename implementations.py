import numpy as np
from costs import compute_loss, compute_loss_log_reg
from gradients import compute_gradient, compute_gradient_log_reg
from helpers import batch_iter


# Gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # initializing the weights
    w = initial_w
    for i in range(max_iters):
        # computing the gradient
        gradient = compute_gradient(y, tx, w)
        # updating the weights
        w = w - gamma * gradient
    # return w with the corresponding loss 
    return w, compute_loss(y, tx, w)


# Stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # initializing the weights, the batch size and the number of batches
    w = initial_w
    batch_size = 1
    num_batches = 1
    for i in range(max_iters):
        # iterating for each batch
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches):
            # computing the gradient
            gradient = compute_gradient(y_batch, tx_batch, w)
            # updating the weights
            w = w - gamma * gradient
    # return w with the corresponding loss 
    return w, compute_loss(y, tx, w)

# Least squares
def least_squares(y, tx):
    # computing the weights by using the formula
    w = np.dot(np.dot(np.linalg.inv(tx.T.dot(tx)), tx.T), y)
    # return w with the corresponding loss 
    return w, compute_loss(y, tx, w)


# Ridge regression
def ridge_regression(y, tx, lambda_):
    # computing the weights by using the formula
    lambda_prime = 2 * tx.shape[0] * lambda_ 
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_ * np.identity(tx.shape[1])), tx.T), y)
    # return w with the corresponding loss 
    return w, compute_loss(y, tx, w)


# Logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # initializing the weights
    w = initial_w
    for iter in range(max_iters):
        # computing the gradient for logistic regression
        gradient = compute_gradient_log_reg(y, tx, w)
        # updating the weights
        w = w - gamma * gradient
    # return w with the corresponding loss
    return w, compute_loss_log_reg(y, tx, w)


# Penalized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # initializing the weights
    w = initial_w
    for iter in range(max_iters):
        # computing the gradient for logistic regression
        gradient = compute_gradient_log_reg(y, tx, w) + (lambda_ / len(y)) * w
        # updating the weights
        w = w - gamma * gradient
    # return w with the corresponding loss and the regularizing term
    return w, compute_loss_log_reg(y, tx, w) + (1/2) * lambda_ * (np.linalg.norm(w)**2)
