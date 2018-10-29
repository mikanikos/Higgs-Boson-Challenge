import numpy as np
from helpers import sigmoid

### Loss computing functions

# Mean Square Error
def compute_mse(e):
    return 1/2*np.mean(e**2)

# Root Mean Square Error
def compute_rmse(e):
    return np.sqrt(2 * compute_mse(e))

# Mean Absolute Error
def compute_mae(e):
    return np.mean(np.abs(e))


# Computing loss for logistic regression
def compute_loss_log_reg(y, tx, w):
    return -(y.T.dot(np.log(sigmoid(tx.dot(w)))) + (1 - y).T.dot(np.log(1 - sigmoid(tx.dot(w)))))


# Computing loss with a selected cost function
def compute_loss(y, tx, w, func="mse"):
    # Computing error
    error = y - tx.dot(w)

    # Using Mean square error
    if func == "mse":
        return compute_mse(error)
    
    # Using Mean absolute error
    elif func == "mae":
        return compute_mae(error)
    
    # Using Root mean square error
    elif func == "rmse":
        return compute_rmse(error)