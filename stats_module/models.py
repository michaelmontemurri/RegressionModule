import numpy as np
import pandas as pd
from sklearn import NotFittedError
from utils import *


class OLS:
    def __init__(self, include_intercept=True):
        self.include_intercept = include_intercept
        self.beta = None

    #should we make use_gradient_descent a parameter automatically use it if p>threshold?
    def fit(self, X, y, use_gradient_descent=False):
        '''
        Fit the OLS model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series

        Returns
        -------
        None
        '''

        #validate data
        validate_data(X, y)

        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        if use_gradient_descent:
            #use gradient descent to find beta
            #will implement this logic later
            pass
        # Check for multicollinearity and singular matrix issues
        try:
            self.beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y

        except np.linalg.LinAlgError:
            raise Warning("feature matrix is singular or nearly singular, "
                          "check for highly correlated features.")
        
        
    def predict(self, X):
        if self.beta is None:
            raise NotFittedError("This OLS instance is not fitted yet. "
                                   "Call 'fit' with appropriate data before using this estimator.")
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        return X_ @ self.beta
    

    def estimate_variance(self, X, y):
        # estimate variance of beta_hat
        #maybe we should have them pass y hat instead of calculating it here.
        y_hat = self.predict(X)
        return sigma_hat_corr(X, y, y_hat)
    
    #function to calculate the leverage of each observation
    def leverages(self, X):
        if self.beta is None:
            raise NotFittedError("This OLS instance is not fitted yet. "
                                   "Call 'fit' with appropriate data before using this estimator.")
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X
        h = X_ @ np.linalg.inv(X_.T @ X_) @ X_.T
        return np.diag(h)
    
    def residuals(self, X, y):
        y_hat = self.predict(X)
        return y - y_hat
    
    def summary(self, X, y):
        y_hat = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_hat)**2)
        r_squared = 1 - ss_res/ss_total
        return {'coefficients': self.beta, 'r_squared': r_squared}
    

class GLS:
    def __init__(self, include_intercept=True):
        self.include_intercept = include_intercept
        self.beta = None
        self.sigma = None

    def fit(self, X, y, sigma):
        '''
        Fit the GLS model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
        sigma : numpy array
            The weight matrix.

        Returns
        -------
        None
        '''
        #validate data
        validate_data(X, y)
            
        if isinstance(sigma, pd.DataFrame):
            sigma = sigma.values

        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        # Check for multicollinearity and singular matrix issues
        try:
            self.beta = np.linalg.inv(X_.T @ np.linalg.inv(sigma) @ X_) @ X_.T @ np.linalg.inv(sigma) @ y
        except np.linalg.LinAlgError:
            raise Warning("feature matrix is singular or nearly singular, "
                          "check for highly correlated features.")
        
        self.sigma = sigma
        
    def predict(self, X):
        if self.beta is None:
            raise NotFittedError("This GLS instance is not fitted yet. "
                                    "Call 'fit' with appropriate data before using this estimator.")
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        return X_ @ self.beta

    def residuals(self, X, y):
        y_hat = self.predict(X)
        return y - y_hat

    def summary(self, X, y):
        y_hat = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_hat)**2)
        r_squared = 1 - ss_res/ss_total
        return {'coefficients': self.beta, 'r_squared': r_squared}


    
            