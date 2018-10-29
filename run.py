import numpy as np

from implementations import *
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from data_processing import process_data, build_poly, clean_data, expand_data
from cross_validation import best_model_selection, ridge_trials

print("Loading data\n")

# Loading data from csv files
y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")
y_te, tx_te, ids_te = load_csv_data("data/test.csv")

# Hyper-parameters definitions
degree = 12
lambda_ =  1e-08

#tx_tr, tx_te = process_data(tx_tr, tx_te, y_tr, y_te)

# Cleaning data from missing values
tx_tr = clean_data(tx_tr)
tx_te = clean_data(tx_te)

# Using cross validation in order to find the best lambda and degree for ridge_regression
print("Cross validation\n")
degree, lambda_ = best_model_selection(tx_tr, y_tr, np.arange(12,13), 2, np.logspace(-10, -7, 3))
print("Best degree: ", degree)
print("Best lambda: ", lambda_, "\n")

tx_tr, tx_te = expand_data(degree, tx_tr, tx_te)

# Preprocessing data: cleaning, standardazing and adding constant column
#tx_tr, tx_te = process_data(tx_tr, tx_te, y_tr, y_te)

# Feature augmentation through polynomials
#tx_tr = build_poly(tx_tr, degree)
#tx_te = build_poly(tx_te, degree)

# Training with ridge regression
print("Training the model\n")
#weights, _ = ridge_regression(y_tr, tx_tr, lambda_)

weights, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w=np.zeros(tx_tr.shape[1]), max_iters=100, gamma=0.1)

# Computing prediction vector
y_pred = predict_labels(weights, tx_te)

# Creating file for submission
create_csv_submission(ids_te, y_pred, "prediction.csv")

print("Done")
