import numpy as np
from scipy.stats import t, f
from stats_module.models import *
from stats_module.loss_estimation import *


# class LinearModelTester:
#     def __init__(self, model):
#         self.model = model
#         if self.model.beta is None:
#             raise ValueError("This model instance is not fitted yet. "
#                                     "Call 'fit' with appropriate data before using this estimator.")


def hypothesis_t_test(model, X, y, null_hypothesis, alpha=0.05):
    '''
    Perform a t-test for individual coefficients.
    Params:
        - X: The feature matrix.
        - y: The response vector.
        - null_hypothesis: The null hypothesis to be tested.
        - alpha: The significance level.
    Returns:
        - A list of dictionaries containing the coefficient, beta estimate, null value, t-statistic, p-value, and whether to reject the null hypothesis.
    '''

    y_hat = model.predict(X)

    n, p = X.shape

    if model.include_intercept:
        p += 1
        X_ = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_ = X

    sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
    
    var_beta_hat = sigma_hat_corr_squared* np.linalg.inv(X_.T @ X_)

    results = []
    for j, (beta_j, b_j, var_j) in enumerate(zip(model.beta, null_hypothesis, np.diag(var_beta_hat))):
        standard_error = np.sqrt(var_j)
        t_stat = (beta_j - b_j) / standard_error
        p_value = 2 * (1 - t.cdf(abs(t_stat), df=n - p))
        reject_flag = p_value < alpha
        results.append({
            'coefficient': j,
            'beta_estimate': beta_j,
            'null_value': b_j,
            't_stat': t_stat,
            'p_value': p_value,
            'reject_null': reject_flag
        })
    return results

def hypothesis_F_test(model, X, y, R, r, alpha=0.05):
    '''
    Perform an F-test for linear hypotheses Rbeta=r.
    Params:
        - X: The feature matrix.
        - y: The response vector.
        - R: The matrix of coefficients.
        - r: The vector of constants.
        - alpha: The significance level.
    Returns:
        - A dictionary containing the F-statistic, p-value, and whether to reject the null hypothesis
    '''
    y_hat = model.predict(X)

    n, p = X.shape

    if model.include_intercept:
        p += 1
        X_ = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_ = X

    sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)

    R_beta_r = (R @ model.beta.T - r)
    F_stat = (R_beta_r.T @ np.linalg.inv(R @ np.linalg.inv(X_.T @ X_) @ R.T) @ R_beta_r / R.shape[0]) / sigma_hat_corr_squared
    p_value = 1 - f.cdf(F_stat, R.shape[0], n-p)
    reject_flag = p_value < alpha
    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'reject_null': reject_flag
    }

def confidence_interval(model, X, y,  alpha=0.05):
    '''
    Construct confidence intervals for coefficients.
    '''
    y_hat = model.predict(X)

    n, p = X.shape

    if model.include_intercept:
        p += 1
        X_ = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_ = X

    sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
    
    var_beta_hat = sigma_hat_corr_squared* np.linalg.inv(X_.T @ X_)

    t_crit = t.ppf(1 - alpha/2, n-p)
    intervals = []
    for i, beta_hat_j in enumerate(model.beta):
        margin = t_crit * np.sqrt(var_beta_hat[i,i])
        intervals.append({
            'coefficient': i,
            'beta_estimate': beta_hat_j,
            'confidence_lower': beta_hat_j - margin, 
            'confidence_upper': beta_hat_j + margin
            }) 
    return intervals

def prediction_interval_m(model, X, y, x_new, alpha=0.05):
    '''
    Construct prediction intervals for a new observation.
    '''
    y_hat = model.predict(X)

    n, p = X.shape

    if model.include_intercept:
        p += 1
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        x_new = np.insert(x_new, 0, 1)
    else:
        X_ = X

    sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
    h_xx = x_new @ np.linalg.inv(X_.T @ X_) @ x_new.T

    m_hat_x_new = x_new @ model.beta
    t_crit = t.ppf(1 - alpha/2, n-p)
    margin = t_crit * np.sqrt(sigma_hat_corr_squared*h_xx) 

    return {
        'mx_new_estimate': m_hat_x_new,
            'confidence_lower': m_hat_x_new - margin, 
            'confidence_upper': m_hat_x_new + margin
            }


def prediction_interval_y(model, X, y, x_new, alpha=0.05):
    '''
    Construct prediction intervals for a new observation.
    '''
    y_hat = model.predict(X)

    n, p = X.shape

    if model.include_intercept:
        p += 1
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        x_new = np.insert(x_new, 0, 1)
    else:
        X_ = X

    sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
    h_xx = x_new @ np.linalg.inv(X_.T @ X_) @ x_new.T

    m_hat_x_new = x_new @ model.beta
    t_crit = t.ppf(1 - alpha/2, n-p)
    margin = t_crit * np.sqrt(sigma_hat_corr_squared*(1+h_xx)) 

    return {
        'mx_new_estimate': m_hat_x_new,
            'confidence_lower': m_hat_x_new - margin, 
            'confidence_upper': m_hat_x_new + margin
            }

    
def model_selection(X, y, models, criterion='naive'):
    """
        Perform model selection based on a specified criterion.

        Parameters
        ----------
        X : Feature matrix.
        y : Response vector.
        models : list
            A list of model instances to be evaluated.
        criterion : str, default='f_stat'
            The criterion for model selection. Options are 'naive', 'train_test', 'loo', 'f_stat'.

        Returns
        -------
        best_model : object
            The model instance that performs the best based on the specified criterion.
    """
    if criterion not in ['naive', 'train_test', 'loo', 'f_stat']:
        raise ValueError("Invalid criterion. Choose from 'naive', 'train_test', 'loo', 'f_stat'.")

    best_model = None
    best_score = np.inf

    for model in models:
        if criterion == 'naive':
            score = naive_loss_estimation(model, X, y)
        elif criterion == 'train_test':
            # Split data into training and testing sets
            n = len(y)
            test_size = int(0.1 * n)  
            train_size = n - test_size  
            train_indices = np.arange(train_size)  # First 90% as training indices
            test_indices = np.arange(train_size, n)  # Last 10% as testing indices
            score = train_test_loss_estimation(model, X, y, train_indices, test_indices)
        elif criterion == 'loo':
            score = loo_loss_estimation(model, X, y)
        
        if score < best_score:
            best_score = score
            best_model = model

    return best_model

def nested_model_selection_f_test(X, y, full_model, reduced_model,alpha = 0.05):
    """
        Perform nested model selection F-test. Null hypothesis is that the reduced model is correct.

        Parameters
        ----------
        X : Feature matrix.
        y : Response vector.
        full_model : object
            Full model instance using all covariates.
        reduced_models : object
            Reduced model instances using only some covariates.
        alpha : Significance of the test: default=0.05

        Returns
        -------

    """

    n, p_full = X.shape
    p_reduced = len(reduced_model.beta)

    y_full_pred = full_model.predict(X)
    y_reduced_pred = reduced_model.predict(X)

    sigma_hat_squared_full = np.sum((y - y_full_pred) ** 2)
    sigma_hat_squared_reduced = np.sum((y - y_reduced_pred) ** 2)

    F_stat = ((sigma_hat_squared_reduced - sigma_hat_squared_full) / (p_full - p_reduced)) / (sigma_hat_squared_full / (n - p_full))
    p_value = 1 - f.cdf(F_stat, p_full - p_reduced, n - p_full)

    reject_flag = p_value < alpha

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'reject_null': reject_flag
    }
        

        








