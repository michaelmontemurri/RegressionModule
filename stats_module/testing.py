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
            - A list of dictionaries containing the coefficient, t-statistic, p-value, and whether to reject the null hypothesis
        '''

        n, p = X.shape
        if self.model.include_intercept:
            p += 1
        
        residuals = self.model.residuals(X, y)
        
        sigma_hat_corr = np.sum(residuals**2)/(n-p)
        if self.model.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        var_beta_hat = sigma_hat_corr * np.linalg.inv(X.T @ X)

        results = []
        for i, (beta_j, b_j, root_var) in enumerate(zip(self.model.beta, null_hypothesis, np.sqrt(np.diag(var_beta_hat)))):
            t_stat = (beta_j - b_j) / root_var
            p_value = 2 * (1 - t.cdf(abs(t_stat), df=n - p))
            reject_flag = p_value < alpha
            results.append({
                'coefficient': i,
                'beta_estimate': beta_j,
                'null_value': b_j,
                't_stat': t_stat,
                'p_value': p_value,
                'reject_null': reject_flag
            })
        return results
    
    def hypotheses_f_test(self, X, y, R, r, alpha=0.05):
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
        n, p = X.shape
        if self.model.include_intercept:
            p += 1
        
        residuals = self.model.residuals(X, y)
        
        sigma_hat_corr = np.sum(residuals**2)/(n-p)
        if self.model.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        var_beta_hat = sigma_hat_corr * np.linalg.inv(X.T @ X)

        R_beta_r = R @ self.model.beta - r

        F_stat = (R_beta_r.T @ np.linalg.inv(R @ np.linalg.inv(X.T @ X) @ R.T) @ R_beta_r / R.shape[0]) / sigma_hat_corr
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
        n, p = X.shape
        if self.model.include_intercept:
            p += 1
        
        residuals = self.model.residuals(X, y)
        sigma_hat_corr = np.sum(residuals**2)/(n-p)
        if self.model.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        var_beta_hat = sigma_hat_corr * np.linalg.inv(X.T @ X)

        t_crit = t.ppf(1 - alpha/2, n-p)
        intervals = []
        for i, beta_hat_j in enumerate(self.model.beta):
            margin = t_crit * np.sqrt(var_beta_hat[i,i])
            intervals.append((beta_hat_j - margin, beta_hat_j + margin))
            
    def confidence_interval_multiple_coefficients_simultaneously(self, beta_hat, se_beta_hat, alpha=0.05):
        '''
        Calculate a confidence interval for multiple coefficients simultaneously.
        '''
    


# Significance testing
#single coeeficient testing
    # t-test critical value tests
    # t-test p-value tests

#multiple coefficient testing
    # F-test critical value tests
    # F-test p-value tests

# general hypotheses RB=r
    #F - test

# Confidence intervals
    # single coefficient
    # multiple coefficients
    # multiple coefficients simultaneously
    # for m(x_new) and y_new


