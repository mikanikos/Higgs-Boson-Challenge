import numpy as np

# Cost functions

def compute_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


# Loss computing functions

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return compute_mse(e)
    # return calculate_mae(e)


def calculate_loss_log_reg(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = 1.0 / (1 + np.exp(-tx.dot(w)))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


# Gradient computing functions

def compute_gradient(y, tx, w):
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient, error


def calculate_gradient_log_reg(y, tx, w):
    """compute the gradient of loss."""
    pred = 1.0 / (1 + np.exp(-tx.dot(w)))
    grad = tx.T.dot(pred - y)
    return grad
