import numpy as np
from costs_gradients import *
from helpers import batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - gamma * gradient
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 1
    num_batches = 1
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches):
            gradient, error = compute_gradient(y_batch, tx_batch, w)
            loss = compute_mse(error)
            w = w - gamma * gradient
    return w, loss


def least_squares(y, tx):
    w = np.dot(np.dot(np.linalg.inv(tx.T.dot(tx)), tx.T), y)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_ * np.identity(tx.shape[1])), tx.T), y)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        loss = calculate_loss_log_reg(y, tx, w)
        gradient = calculate_gradient_log_reg(y, tx, w)
        w = w - gamma * gradient
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        loss = calculate_loss_log_reg(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_gradient_log_reg(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, loss
