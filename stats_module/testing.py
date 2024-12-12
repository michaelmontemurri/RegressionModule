import numpy as np
from scipy.stats import t, f

class LinearModelTester:
    def __init__(self, model):
        self.model = model
        if self.model.beta is None:
            raise ValueError("This model instance is not fitted yet. "
                                    "Call 'fit' with appropriate data before using this estimator.")


    def hypothesis_t_test(self, X, y, null_hypothesis, alpha=0.05):
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

        y_hat = self.model.predict(X)

        n, p = X.shape

        if self.model.include_intercept:
            p += 1
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
        
        var_beta_hat = sigma_hat_corr_squared* np.linalg.inv(X_.T @ X_)

        results = []
        for j, (beta_j, b_j, var_j) in enumerate(zip(self.model.beta, null_hypothesis, np.diag(var_beta_hat))):
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
    
    def hypothesis_F_test(self, X, y, R, r, alpha=0.05):
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
        y_hat = self.model.predict(X)

        n, p = X.shape

        if self.model.include_intercept:
            p += 1
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)

        R_beta_r = (R @ self.model.beta.T - r)
        F_stat = (R_beta_r.T @ np.linalg.inv(R @ np.linalg.inv(X_.T @ X_) @ R.T) @ R_beta_r / R.shape[0]) / sigma_hat_corr_squared
        p_value = 1 - f.cdf(F_stat, R.shape[0], n-p)
        reject_flag = p_value < alpha
        return {
            'F_stat': F_stat,
            'p_value': p_value,
            'reject_null': reject_flag
        }

    def confidence_interval(self, X, y,  alpha=0.05):
        '''
        Construct confidence intervals for coefficients.
        '''
        y_hat = self.model.predict(X)

        n, p = X.shape

        if self.model.include_intercept:
            p += 1
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
        
        var_beta_hat = sigma_hat_corr_squared* np.linalg.inv(X_.T @ X_)

        t_crit = t.ppf(1 - alpha/2, n-p)
        intervals = []
        for i, beta_hat_j in enumerate(self.model.beta):
            margin = t_crit * np.sqrt(var_beta_hat[i,i])
            intervals.append({
                'coefficient': i,
                'beta_estimate': beta_hat_j,
                'confidence_lower': beta_hat_j - margin, 
                'confidence_upper': beta_hat_j + margin
                }) 
        return intervals
    
    def prediction_interval_m(self, X, y, x_new, alpha=0.05):
        '''
        Construct prediction intervals for a new observation.
        '''
        y_hat = self.model.predict(X)

        n, p = X.shape

        if self.model.include_intercept:
            p += 1
            X_ = np.column_stack([np.ones(X.shape[0]), X])
            x_new = np.insert(x_new, 0, 1)
        else:
            X_ = X

        sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
        h_xx = x_new @ np.linalg.inv(X_.T @ X_) @ x_new.T

        m_hat_x_new = x_new @ self.model.beta
        t_crit = t.ppf(1 - alpha/2, n-p)
        margin = t_crit * np.sqrt(sigma_hat_corr_squared*h_xx) 

        return {
            'mx_new_estimate': m_hat_x_new,
                'confidence_lower': m_hat_x_new - margin, 
                'confidence_upper': m_hat_x_new + margin
                }
    

    def prediction_interval_y(self, X, y, x_new, alpha=0.05):
        '''
        Construct prediction intervals for a new observation.
        '''
        y_hat = self.model.predict(X)

        n, p = X.shape

        if self.model.include_intercept:
            p += 1
            X_ = np.column_stack([np.ones(X.shape[0]), X])
            x_new = np.insert(x_new, 0, 1)
        else:
            X_ = X

        sigma_hat_corr_squared = np.sum((y-y_hat)**2) / (n - p)
        h_xx = x_new @ np.linalg.inv(X_.T @ X_) @ x_new.T

        m_hat_x_new = x_new @ self.model.beta
        t_crit = t.ppf(1 - alpha/2, n-p)
        margin = t_crit * np.sqrt(sigma_hat_corr_squared*(1+h_xx)) 

        return {
            'mx_new_estimate': m_hat_x_new,
                'confidence_lower': m_hat_x_new - margin, 
                'confidence_upper': m_hat_x_new + margin
                }

    
  


