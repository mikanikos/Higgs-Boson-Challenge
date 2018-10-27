import numpy as np
import pandas as pd

from implementations import *
from proj1_helpers import *
from helpers import *
from cross_validation import *

# LOAD DATA
y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")
y_te, tx_te, ids_te = load_csv_data("data/test.csv")


# HYPERPARAMETERS DEFINITION
degree = 1
lambda_ =  0.01 #0.0016681005372000592
max_iters = 50
gamma = 0.1

# USING CROSS VALIDATION TO FIND THE BEST DEGREE AND LAMBDA
degree, lambda_ = best_model_selection(tx_tr, y_tr, np.arange(7,8), 5, np.logspace(0, 5, 50))

print(degree, lambda_)

tx_tr, tx_te = preprocess_data(tx_tr, tx_te, y_tr, y_te)

w = np.zeros(tx_tr.shape[1])
#w = np.zeros((tx_tr.shape[1],1))

# DATA AUGMENTATION WITH POLY
#tx_tr = build_poly(tx_tr, degree)
#tx_te = build_poly(tx_te, degree)

# TRAINING FUNCTIONS
#weights, _ = least_squares_GD(y_tr, tx_tr, w, max_iters, gamma)
#weights, _ = least_squares_SGD(y_tr, tx_tr, w, max_iters, gamma)
#weights, _ = least_squares(y_tr, tx_tr)
weights, _ = ridge_regression(y_tr, tx_tr, lambda_)
#weights, _ = logistic_regression(y_tr, tx_tr, w, max_iters, gamma)
#weights, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, w, max_iters, gamma)

# COMPUTE PREDICTION VECTOR AND ACCURACY
y_pred = predict_labels(weights, tx_te)

print(compute_accuracy(y_te, y_pred))

# CREATING FILE FOR SUBMISSION
create_csv_submission(ids_te, y_pred, "result.csv")
