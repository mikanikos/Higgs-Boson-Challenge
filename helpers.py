import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def standardize(tx):
    """Standardize the original data set."""
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def compute_accuracy(y_true, y_pred):
    return sum(np.array(y_pred) == np.array(y_true)) / float(len(y_true))


def clean_data(x):
    x[x == -999] = np.nan
    median_x = np.nanmean(x, axis=0)
    return np.where(np.isnan(x), median_x, x)

def add_constants(x, y):
    return np.c_[np.ones((y.shape[0], 1)), x]

def preprocess_data(tx_tr, tx_te, y_tr, y_te):
    
    #stds = np.std(tx_tr, axis=0)
    #deleted_cols_ids = np.where(stds == 0)

    #tx_tr = np.delete(tx_tr, deleted_cols_ids, axis=1)
    #tx_te = np.delete(tx_te, deleted_cols_ids, axis=1)

    tx_tr = clean_data(tx_tr)
    tx_te = clean_data(tx_te)

    tx_tr = standardize(tx_tr) 
    tx_te = standardize(tx_te)

    tx_tr = add_constants(tx_tr, y_tr)
    tx_te = add_constants(tx_te, y_te)

    return tx_tr, tx_te