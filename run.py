import numpy as np

from implementations import ridge_regression
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from data_processing import process_data, build_poly

print("Loading data\n")

# Loading data from csv files
y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")
y_te, tx_te, ids_te = load_csv_data("data/test.csv")

# Hyper-parameters definitions
degree = 7
lambda_ = 0.00025

# Preprocessing data: cleaning, standardazing and adding constant column
tx_tr, tx_te = process_data(tx_tr, tx_te, y_tr, y_te)

# Feature augmentation through polynomials
tx_tr = build_poly(tx_tr, degree)
tx_te = build_poly(tx_te, degree)

# Training with ridge regression
print("Training the model\n")
weights, _ = ridge_regression(y_tr, tx_tr, lambda_)

# Computing prediction vector
y_pred = predict_labels(weights, tx_te)

# Creating file for submission
create_csv_submission(ids_te, y_pred, "prediction.csv")

print("Done")
