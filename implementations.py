import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from helpers import *
from costs import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - gamma * gradient
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            gradient, error = compute_gradient(y_batch, tx_batch, w)
            loss = compute_mse(error)
            w = w - gamma * gradient
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=1):
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            gradient, loss = compute_gradient(y_batch, tx_batch, w, func="logistic")
            w = w - gamma * gradient
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ## TO IMPLEMENT
    return

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    I = lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + I, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
