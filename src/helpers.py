"""some helper functions."""

import numpy as np

def fix_null(tx):
    for feature in range(tx.shape[1]):
        row_indices = np.where(tx[:,feature] == -999.0)[0]
        clean_data = [x for x in tx[:,feature] if x != -999.0]
        #clean_data = np.mean(tx[np.where(tx_predict[:, 0] != -999)[0], 0])
        mean = np.mean(clean_data)
        # print(f"feature {feature} mean:{mean}\n")
        # print(row_indices)
        # print("\n")
        tx[row_indices,feature] = mean
    return tx


def standardize(x):
    # Standardize the original data set.
    # standardize the data into [-1,1]
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def detect_outliers(tx):
    outlier_indices = np.array([]) 
    for feature in range(tx.shape[1]):
        _, mean, std = standardize(tx[:,feature])
        #row_indices = np.where(np.absolute(tx[:,feature]-mean) > 3*std)[0]
        row_indices_big = np.where(tx[:,feature]-mean > 3*std)[0]
        row_indices_small = np.where(mean - tx[:,feature] > 3*std)[0]
        tx[row_indices_big] = mean + 3*std
        tx[row_indices_small] = mean - 3*std
    return tx

def expand_dimen(tx,k): # (x1, x2, x3 ...) -> (x1, x1^2, x^k)
    tx_base = tx
    for i in range(2, k+1):
        tx = np.hstack((tx, np.power(tx_base, i)))
    return tx

def feature_aug(tx, k):
    tx_new = expand_dimen(tx, k)
    return tx_new

def standlization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def split_data(x, y, ratio, myseed):
    
    """split the train dataset to train and validation dataset based on the split ratio."""
    
    # set seed
    # ratio = 0.75
    np.random.seed(myseed)
    # generate random indices
    num_row = len(x)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_train = indices[: index_split]
    index_val = indices[index_split:]
    # create split
    x_tr = x[index_train]
    x_val = x[index_val]
    y_tr = y[index_train]
    y_val = y[index_val]
    
    return x_tr, x_val, y_tr, y_val

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - (np.mean(np.abs(Y_pred - Y_label))/2)
    return acc

def index_search(tx):
    PRI_jet_num_0 = np.where(tx[:,22] == 0)[0]
    PRI_jet_num_1 = np.where(tx[:,22] == 1)[0]
    PRI_jet_num_2 = np.where(tx[:,22] == 2)[0]
    PRI_jet_num_3 = np.where(tx[:,22] == 3)[0]
    return PRI_jet_num_0, PRI_jet_num_1, PRI_jet_num_2, PRI_jet_num_3
    # return ([tx[PRI_jet_num_0,:], tx[PRI_jet_num_1,:], tx[PRI_jet_num_2,:], tx[PRI_jet_num_3,:]],
      #       [y[PRI_jet_num_0], y[PRI_jet_num_1], y[PRI_jet_num_2], y[PRI_jet_num_3]])

def divide(PRI_jet_num_0, PRI_jet_num_1, PRI_jet_num_2, PRI_jet_num_3, tx, y):
    return ([tx[PRI_jet_num_0,:], tx[PRI_jet_num_1,:], tx[PRI_jet_num_2,:], tx[PRI_jet_num_3,:]],
             [y[PRI_jet_num_0], y[PRI_jet_num_1], y[PRI_jet_num_2], y[PRI_jet_num_3]])

def ridge_regression_demo_new(x, y):
    """ridge regression demo."""
    # define parameter
    lambdas = 0.001
    # split data
    # ridge regression with different lambda
    weight = ridge_regression(y, x, lambdas)
    return weight      

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)