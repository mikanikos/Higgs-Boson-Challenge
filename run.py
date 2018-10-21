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
degree = 8
lambda_ =  0.0016681005372000592 #0.0016681005372000592
max_iters = 500 
gamma = 0.01

# USING CROSS VALIDATION TO FIND THE BEST DEGREE AND LAMBDA
#degree = best_degree_selection(tx_tr, y_tr, np.arange(2,9), 3, np.logspace(-5, 0, 10))
#print(degree)

# PREPROCESS DATA
tx_tr = clean_data(tx_tr)
tx_te = clean_data(tx_te)

tx_tr, _, _ = standardize(tx_tr)
tx_te, _, _ = standardize(tx_te)

y_tr, tx_tr = build_model_data(tx_tr, y_tr)
y_te, tx_te = build_model_data(tx_te, y_te)

w = np.zeros(tx_tr.shape[1])

# DATA AUGMENTATION WITH POLY
tx_tr = build_poly(tx_tr, degree)
tx_te = build_poly(tx_te, degree)

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
