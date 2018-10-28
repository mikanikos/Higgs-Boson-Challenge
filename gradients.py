def compute_gradient(y, tx, w, func="mse"):
    if func == "mse":
        return compute_mse_gradient(y, tx, w)
    elif func == "logistic":
        return compute_logistic_gradient(y, tx, w)

def compute_mse_gradient(y, tx, w):
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient, error

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    loss = compute_loss(y, tx, w, func="logistic")
    return grad, loss