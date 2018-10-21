from implementations import ridge_regression
from helpers import *
from costs_gradients import *
from proj1_helpers import *


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    tx_tr = clean_data(x_tr)
    tx_te = clean_data(x_te)

    tx_tr, _, _ = standardize(tx_tr)
    tx_te, _, _ = standardize(tx_te)

    y_tr, tx_tr = build_model_data(tx_tr, y_tr)
    y_te, tx_te = build_model_data(tx_te, y_te)

    # form data with polynomial degree
    tx_tr = build_poly(tx_tr, degree)
    tx_te = build_poly(tx_te, degree)
    
    print("Test: d = ", degree, "; l = ", lambda_)

    # ridge regression
    w, _ = ridge_regression(y_tr, tx_tr, lambda_)

    y_pred = predict_labels(w, tx_te)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w))
    
    print("Accuracy = ", compute_accuracy(y_pred, y_te), "; loss = ", loss_te)
    print()

    return loss_tr, loss_te, w


def best_degree_selection(x, y, degrees, k_fold, lambdas, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te,_ = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree]
