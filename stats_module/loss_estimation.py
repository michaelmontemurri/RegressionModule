import numpy as np
from stats_module.utils import *
import copy

def naive_loss_estimation(model, X, y):
    """
    Calculates the naive loss estimator for a given model over a given dataset.

    Parameters:
        - model:    The model which the naive loss will be calculated for. Assumes that model.fit() and model.predict() is implemented
        - X:        The co-variates for the data set which the naive loss estimate will be calculated for. Assumes n-p matrix of floats/ints
        - y:        The response corresponding to X. Assumes vector of length n.
    Returns:
        - (float)   The naive loss estimate for model over the dataset (X,y)
    """
    validate_data(X,y)

    model.fit(X,y)
    y_pred = model.predict(X)
    naive_loss_estimate = np.mean((y-y_pred)**2) 

    return naive_loss_estimate



def train_test_loss_estimation(model, X, y, train_range, test_range):
    """
    Calculates the training/testing loss estimator for a given model over a given dataset.

    Parameters:
        - model:        The model which the training/testing loss will be calculated for. Assumes that model.fit() and model.predict() are implemented.
        - X:            The co-variates for the data set which the naive loss estimate will be calculated for. Assumes n-p matrix of floats/ints
        - y:            The response corresponding to X. Assumes vector of length n.
        - train_range:  A list of indices which will be used for training
        - test_range:   A list of indices which will be used for calculating the loss estimate
    Returns:
        - (float)   The naive loss for model over the dataset (X,y)
    """


    validate_data(X, y)

    X_train = X[train_range]
    y_train = y[train_range]


    X_test = X[test_range]
    y_test = y[test_range]

    training_model = copy.deepcopy(model)
    training_model.fit(X_train, y_train)

    return np.mean( (y_test - training_model.predict(X_test))**2 )



    X_train = X[train_range]
    y_train = y[train_range]



    training_model = copy.deepcopy(model)
    training_model.fit(X_train, y_train)

    y_pred_test = training_model.predict(X_test)
    train_test_loss_estimate = np.mean((y_test-y_pred_test)**2)

    return train_test_loss_estimate





def loo_loss_estimation(model, X, y):
    """
    Calculates the leave-one-out loss estimator for a given model over a given dataset. Assuming a linear regression model for closed form solution

    Parameters:
        - model:    The model which the leave-one-out loss will be calculated for. Assumes that model.fit() and model.predict() is implemented
        - X:        The co-variates for the data set which the leave-one-out loss estimate will be calculated for. Assumes n-p matrix of floats/ints
        - y:        The response corresponding to X. Assumes vector of length n.
    Returns:
        - (float)   The leave-one-out loss estimate for model over the dataset (X,y)
    """
    validate_data(X,y)
    


    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)

    y_hat = H @ y
    residuals = y - y_hat

    return np.linalg.norm(residuals/(np.ones(len(h))-h)) / X.shape[0]

    



 


