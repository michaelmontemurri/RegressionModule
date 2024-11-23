import numpy as np
import stats_module.utils
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
    stats_module.utils.validate_data(X,y)

    model.fit(X)
    y_pred = model.predict(X)
    naive_loss_estimate = np.mean((y-y_pred)**2)

    return naive_loss_estimate



def train_test_loss_estimation(model, X_train, y_train, X_test, y_test):
    """
    Calculates the training/testing loss estimator for a given model over a given dataset.

    Parameters:
        - model:    The model which the training/testing loss will be calculated for. Assumes that model.fit() and model.predict() are implemented.
        - X_train:  The co-variates for the training set. Assuming n1-p matrix of numbers.  
        - y_train:  The features for the training set. Assuming n1-p matrix of numbers.
        - X_test:   The co-variates for the testing set. Assuming n2-p matrix of numbers.
        - y_test:   The features for the testing set. Assuming n2-p matrix of numbers.
    Returns:
        - (float)   The naive loss for model over the dataset (X,y)
    """
    stats_module.utils.validate_data(X_train, y_train)
    stats_module.utils.validate_data(X_test, y_test)

    training_model = copy.deepcopy(model)
    training_model.fit(X_train, y_train)

    y_pred_test = training_model.predict(X_test)
    train_test_loss_estimate = np.mean((y_test-y_pred_test)**2)

    return train_test_loss_estimate



def loo_loss_estimation(model, X, y):
    """
    Calculates the leave-one-out loss estimator for a given model over a given dataset.

    Parameters:
        - model:    The model which the leave-one-out loss will be calculated for. Assumes that model.fit() and model.predict() is implemented
        - X:        The co-variates for the data set which the leave-one-out loss estimate will be calculated for. Assumes n-p matrix of floats/ints
        - y:        The response corresponding to X. Assumes vector of length n.
    Returns:
        - (float)   The leave-one-out loss estimate for model over the dataset (X,y)
    """
    stats_module.utils.validate_data(X,y)
    
    y_pred = np.empty(y.shape[0])
    for i in range(X.shape[0]):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)

        training_model_i = copy.deepcopy(model)
        training_model_i.fit(X_train, y_train)

        y_pred[i] = training_model_i.predict(X[i])


    loo_loss_estimate = np.mean((y-y_pred)**2)
    return loo_loss_estimate
    

    
    



