import numpy as np
from scipy.stats import t, f
from stats_module.models import *
from stats_module.loss_estimation import *


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
    p_reduced = len(reduced_model.selected_features)
    if full_model.include_intercept:
        p_full += 1
        p_reduced += 1

    y_full_pred = full_model.predict(X)
    y_reduced_pred = reduced_model.predict(X)

    sigma_hat_squared_full = np.sum((y - y_full_pred) ** 2)
    sigma_hat_squared_reduced = np.sum((y - y_reduced_pred) ** 2)

    df1 = p_full - p_reduced
    df2 = n - p_full

    numerator = max((sigma_hat_squared_reduced - sigma_hat_squared_full), 0)

    F_stat = (numerator/df1) / (sigma_hat_squared_full/df2)
    p_value = 1 - f.cdf(F_stat, df1,df2)

    reject_flag = p_value < alpha

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'reject_null': reject_flag
    }