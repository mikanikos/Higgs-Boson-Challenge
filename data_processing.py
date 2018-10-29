import numpy as np


# Cleaning data from -999 values
def clean_data(x):
	## Convert -999 to NaN as we believe these are misidentified data
    ## Recording as NaN prevents them from influencing nanmean calculations
    x[x == -999] = np.nan
    mean_x = np.nanmean(x, axis=0)
    return np.where(np.isnan(x), mean_x, x)


# Standardizing by subtracting the mean and dividing by standard deviation
def standardize(x):
    """Standardize the original data set."""
    means = np.mean(x, axis=0)
    x = x-means
    stds = np.std(x, axis=0)
    # this prevents division by zero
    stds[stds == 0] = 1
    x = x/stds
    return x, means, stds


# Adding 1-column at the data matrix 
def add_constants(x, y):
    return np.c_[np.ones((y.shape[0], 1)), x]


# Preprocessing data function
def process_data(tx_tr, tx_te, y_tr, y_te):
    
    # Cleaning data from -999 values
    tx_tr = clean_data(tx_tr)
    tx_te = clean_data(tx_te)

    # Standardizing data
    tx_tr, mean, std = standardize(tx_tr)
    tx_te = (tx_te-mean)/std

    # Adding constants vector as a first column 
    tx_tr = add_constants(tx_tr, y_tr)
    tx_te = add_constants(tx_te, y_te)

    return tx_tr, tx_te


## Because expansion and standardization are transformations of our initial feature set
## We must apply identical transformations to all feature sets we wish to make predictions upon
def expand_data(degree, tx_tr, tx_te, tx_pred = None):
    ## Extract jet numbers as three indicator variables
    ## Remove them so they will not be standardized or expanded
    jets_tr = jet_nums(tx_tr)
    jets_te= jet_nums(tx_te)
    ## Remove redundant columns
    res_tr = extract_col(tx_tr)
    res_te = extract_col(tx_te)
    ## Expand features to include polynomial terms
    res_tr = build_poly(tx_tr, degree)
    res_te = build_poly(tx_te, degree)
    ## Standardize
    res_tr, mean, std = standardize(res_tr)
    res_te = (res_te-mean)/std
    ## Fix NaNs resulting from division by 0
    res_tr[np.isnan(res_tr)]=1
    res_te[np.isnan(res_te)]=1
    ## Reconcatenate jet indicator columns
    res_tr = np.c_[res_tr, jets_tr]
    res_te = np.c_[res_te, jets_te]
    return res_tr, res_te #, res_pred


## Jet number seems to be categorical, taking on three discrete values
## Relative values do not seem to have meaning, so coefficients are not a good way to treat this
## Solution: Split this into three indicator vectors. Each indicator takes a different coefficient
def jet_nums(tx):
    jets = tx[:,22]
    new_tx = np.delete(tx, 22, axis=1)
    jet0 = np.zeros((jets.shape[0],1))
    jet0[jets==0] = 1
    jet1 = np.zeros((jets.shape[0],1))
    jet1[jets==1] = 1
    jet2 = np.zeros((jets.shape[0],1))
    jet2[jets==2] = 1
    jet3 = np.zeros((jets.shape[0],1))
    jet3[jets==3] = 1
    result = np.c_[jet0, jet1, jet2, jet3]
    return result


# Extract 22th column from the data
def extract_col(tx):
    result = np.delete(tx, 22, axis=1)
    return result


# Adding polynomial features up to the selected degree
def build_poly(x, degree):
    poly_x = np.ones((len(x), 1))
    for d in range(1, degree+1):
        poly_x = np.c_[poly_x, np.power(x, d)]
    return poly_x