import numpy as np

def compute_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_logistic_cost(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_loss(y, tx, w, func="mse"):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    if func == "mse":
        return compute_mse(e)
    elif func == "mae":
        return compute_mae(e)
    elif func == "logistic":
        return compute_logistic_cost(y, tx, w)
