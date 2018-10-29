import numpy as np

from implementations import *
from helpers import compute_accuracy, split_data
from data_processing import process_data, build_poly, expand_data, add_constants, clean_data
from costs import compute_loss, compute_loss_log_reg
from proj1_helpers import predict_labels


# Building k indices for cross validation
def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
    

# Cross validation
def cross_validation(y, x, k_indices, k, lambda_, degree):
    
    # Dividing in subgroups
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    tx_te = x[te_indice]
    tx_tr = x[tr_indice]

    #tx_tr, tx_te = expand_data(degree, tx_tr, tx_te)

    # Preprocessing data: cleaning, standardazing and adding constant column
    tx_tr, tx_te = process_data(tx_tr, tx_te, y_tr, y_te)

    # Feature augmentation through polynomials
    tx_tr = build_poly(tx_tr, degree)
    tx_te = build_poly(tx_te, degree)

    # Printing degree and lambda tested
    print("Test: d = ", degree, "; l = ", lambda_)

    # Training with ridge regression
    #w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w=np.zeros(tx_tr.shape[1]), max_iters=30, gamma=0.001)
    #w, _ = least_squares_GD(y_tr, tx_tr, initial_w=np.zeros(tx_tr.shape[1]), max_iters=50, gamma=0.1)
    #w, _ = least_squares_SGD(y_tr, tx_tr, initial_w=np.zeros(tx_tr.shape[1]), max_iters=100, gamma=0.01)
    #w, _ = least_squares(y_tr, tx_tr)
    w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    #w, _ = logistic_regression(y_tr, tx_tr, initial_w=np.zeros(tx_tr.shape[1]), max_iters=30, gamma=0.001)
    #w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w=np.zeros(tx_tr.shape[1]), max_iters=30, gamma=0.1)

    # Computing prediction vector
    y_pred = predict_labels(w, tx_te)
    
    # Computing loss and accuracy on test set 
    loss_te = compute_loss(y_te, tx_te, w)
    accuracy = compute_accuracy(y_te, y_pred)

    print("Accuracy = ", accuracy, "; loss = ", loss_te, "\n")

    return loss_te, accuracy


# Method to select the best hyper-parameters for a model
def best_model_selection(x, y, degrees, k_fold, lambdas, seed = 10):
    # Splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Iterating over degrees and lambdas
    best_lambdas = []
    best_losses = []
    best_acc = []
    # For each degree
    for degree in degrees:
        losses_te = []
        acc_te = []
        # For each lambda
        for lambda_ in lambdas:
            losses_te_tmp = []
            acc_te_tmp = []
            # For each split
            for k in range(k_fold):
                # Using cross validation for each degree and lambda
                loss_te, acc = cross_validation(y, x, k_indices, k, lambda_, degree)
                # Saving accuracy and loss 
                losses_te_tmp.append(loss_te)
                acc_te_tmp.append(acc)

            # Taking the mean of loss and accuracy for the cross validation iteration
            losses_te.append(np.mean(losses_te_tmp))
            acc_te.append(np.mean(acc_te_tmp))
        
        # Selecting the best parameters for maximizing the accuracy
        ind_lambda_opt = np.argmax(acc_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_acc.append(acc_te[ind_lambda_opt])
        
    ind_best_degree = np.argmax(best_acc)      
    print("Best accuracy: ", max(best_acc))

    return degrees[ind_best_degree], best_lambdas[ind_best_degree]


# Ridge regression trials
def ridge_trials(y, tx, tx_sub, degree_range, lambda_range, partitions=2):
    ## Split data into test and training sets
    ## If partitions > 2, use k-fold cross-validation
    glob_tx_tr, glob_tx_te, glob_y_tr, glob_y_te = split_data(tx, y, 0.8)

    ## Initial results: losses, weights, preditions and (test) losses
    models = []
    losses = []
    accuracies = []
    predictions = []
    
    ## Loops over range of degrees
    degrees = range(degree_range[0], degree_range[1])
    lambdas = np.logspace(lambda_range[0], lambda_range[1], num=1+(lambda_range[1]-lambda_range[0]))
    for degree in degrees:
        ## Loops over range of lambdas
        for lambda_ in lambdas:
            print("Trying degree", degree,"with lambda =", lambda_,":")

            tx_tr, tx_te, tx_pred = expand(degree, glob_tx_tr, glob_tx_te, tx_sub)

            w, loss = ridge_regression(glob_y_tr, tx_tr, lambda_)
            print("\tTraining Loss = ", loss)

            y_test = predict_labels(w, tx_te)
            test_loss = compute_loss(glob_y_te, tx_te, w)
            accuracy = compute_accuracy((y_test+1)/2, glob_y_te)
            y_pred = predict_labels(w, tx_pred)

            print("\tTest Loss = ", test_loss, " Test Accuracy = ", accuracy )
            models.append(("ridge_regression", degree, lambda_, w))
            losses.append(test_loss)
            accuracies.append(accuracy)
            predictions.append(y_pred)
    return models, losses, accuracies, predictions
    
MAX_ITERS = 100  
GAMMA = 0.6

## Performs logistic trials over set of hyper-parameters (degrees)
## Results result from these trials with corresponding test losses
def logistic_trials(y, tx, tx_sub, degree_range, partitions=2):
    ## Split data into test and training sets
    ## If partitions > 2, use k-fold cross-validation
    glob_tx_tr, glob_tx_te, glob_y_tr, glob_y_te = split_data(tx, y, 0.8)

    ## Initial results: losses, weights, preditions and (test) losses
    models = []
    losses = []
    accuracies = []
    predictions = []
    
    ## Loops over range of degrees
    degrees = range(degree_range[0], degree_range[1])
    for degree in degrees:
        print("Trying degree", degree, ":")

        tx_tr, tx_te, tx_pred = expand(degree, glob_tx_tr, glob_tx_te, tx_sub)        
        initial_w = np.ones(tx_tr.shape[1])
        
        w, loss = logistic_regression(glob_y_tr, tx_tr, initial_w, MAX_ITERS, GAMMA)
        print("\tTraining Loss = ", loss)
        
        y_test = predict_labels(w, tx_te)
        test_loss = compute_loss(glob_y_te, tx_te, w, func="logistic")
        accuracy = compute_accuracy((y_test+1)/2, glob_y_te)
        y_pred = predict_labels(w, tx_pred)

        print("\tTest Loss = ", test_loss, " Test Accuracy = ", accuracy )
        models.append(("logistic_SGD", degree, w))
        losses.append(test_loss)
        accuracies.append(accuracy)
        predictions.append(y_pred)
    return models, losses, accuracies, predictions
