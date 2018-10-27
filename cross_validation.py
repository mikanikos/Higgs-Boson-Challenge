from implementations import reg_logistic_regression, ridge_regression
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
    tx_te = x[te_indice]
    tx_tr = x[tr_indice]
    
    tx_tr, tx_te = preprocess_data(tx_tr, tx_te, y_tr, y_te)

    # form data with polynomial degree
    tx_tr = build_poly(tx_tr, degree)
    tx_te = build_poly(tx_te, degree)
    
    print("Test: d = ", degree, "; l = ", lambda_)

    max_iters = 50
    gamma = 0.1
    #w = np.zeros((tx_tr.shape[1],1))
    w = np.zeros(tx_tr.shape[1])

    w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    #w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, w, max_iters, gamma)

    y_pred = predict_labels(w, tx_te)

    # calculate the loss for train and test data
    #loss_tr = calculate_loss_log_reg(y_tr, tx_tr, w)  + (1/2) * lambda_ * w.T.dot(w)
    #loss_te = calculate_loss_log_reg(y_te, tx_te, w)
    
    loss_te = compute_loss(y_te, tx_te, w)
    accuracy = compute_accuracy(y_te, y_pred)

    print("Accuracy = ", accuracy, "; loss = ", loss_te)
    print()

    return loss_te, accuracy


def best_model_selection(x, y, degrees, k_fold, lambdas, seed = 10):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_losses = []
    best_acc = []
    #vary degree
    for degree in degrees:
        # cross validation
        losses_te = []
        acc_te = []
        for lambda_ in lambdas:
            losses_te_tmp = []
            acc_te_tmp = []
            for k in range(k_fold):
                loss_te, acc = cross_validation(y, x, k_indices, k, lambda_, degree)
                losses_te_tmp.append(loss_te)
                acc_te_tmp.append(acc)
            losses_te.append(np.mean(losses_te_tmp))
            acc_te.append(np.mean(acc_te_tmp))
        
        #ind_lambda_opt = np.argmin(losses_te)
        ind_lambda_opt = np.argmax(acc_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        #best_losses.append(losses_te[ind_lambda_opt])
        best_acc.append(acc_te[ind_lambda_opt])
        
    #ind_best_degree =  np.argmin(best_losses)
    ind_best_degree = np.argmax(best_acc)      
        
    return degrees[ind_best_degree], best_lambdas[ind_best_degree]
