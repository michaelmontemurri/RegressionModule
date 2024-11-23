import numpy as np
import pandas as pd

def validate_data(X, y):
    if y.ndim > 1:
        raise ValueError("y must be a 1-dimensional array.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of observations.")

    #check for categorical variables
    if isinstance(X, pd.DataFrame) and X.select_dtypes(include='object').shape[1] > 0:
        raise ValueError("X cannot have categorical variables.")
    
    #check that X and y are all numeric
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be all numeric.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must be numeric.")

    
# sigma_hat_corr^2 estimator.
def sigma_hat_corr(X, y, y_hat):
    n = X.shape[0]
    p = X.shape[1]
    ss_residuals = np.sum((y - y_hat) ** 2)
    return n/(n-p) * ss_residuals
 