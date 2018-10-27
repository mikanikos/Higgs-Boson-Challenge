import numpy as np

# Cost functions

def compute_rmse(e):
    return np.sqrt(2 * compute_mse(e))


def compute_mse(e):
    return 1/2*np.mean(e**2)


def compute_mae(e):
    return np.mean(np.abs(e))


# Loss computing functions

def compute_loss(y, tx, w):
    error = y - tx.dot(w)
    return compute_mse(error)
    # return calculate_mae(e)


def calculate_loss_log_reg(y, tx, w):
    #txw = tx @ w
    #critical_value = np.float64(709.0)
    #overf = np.where(txw >= critical_value)
    #postives = np.sum(txw[overf] - y[overf] * txw[overf])
    #rest_ids = np.where(txw < critical_value)
    #rest = np.sum(np.log(1 + np.exp(txw[rest_ids])) - y[rest_ids] * txw[rest_ids])
    #return rest + postives
    return -(y.T.dot(np.log(sigmoid(tx.dot(w)))) + (1 - y).T.dot(np.log(1 - sigmoid(tx.dot(w)))))


# Gradient computing functions

def compute_gradient(y, tx, w):
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient, error


def calculate_gradient_log_reg(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def sigmoid(t):
    #negative_ids = np.where(t < 0)
    #positive_ids = np.where(t >= 0)
    #t[negative_ids] = np.exp(t[negative_ids]) / (1 + np.exp(t[negative_ids]))
    #t[positive_ids] = 1 / (np.exp(-t[positive_ids]) + 1)
    #return t
    return np.exp(t)/(1.0 + np.exp(t))
