import numpy as np

# sigma_hat_corr^2 estimator.
def sigma_hat_corr(X, y, y_hat):
    n = X.shape[0]
    p = X.shape[1]
    ss_residuals = np.sum((y - y_hat) ** 2)
    return n/(n-p) * ss_residuals
 