import numpy as np
from helpers import sigmoid


### Gradient computing functions

# Computing gradient
def compute_gradient(y, tx, w):
    error = y - tx.dot(w)
    return -tx.T.dot(error) / len(error)


# Computing gradient for logistic regression
def compute_gradient_log_reg(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w)) - y)