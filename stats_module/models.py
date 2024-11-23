import numpy as np
import pandas as pd
from sklearn import NotFittedError

class LinearRegression:
    def __init__(self, include_intercept=True):
        self.include_intercept = include_intercept
        self.beta = None

    def fit(self, X, y):
        if y.ndim > 1:
            raise ValueError("y must be a 1-dimensional array.")

        # handle categorical data by converting categorical columns into dummy variables
        if isinstance(X, pd.DataFrame):
            X = pd.get_dummies(X, drop_first=True)  # drop first to avoid multicollinearity
        
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        # Check for multicollinearity and singular matrix issues
        try:
            self.beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        except np.linalg.LinAlgError:
            raise Warning("feature matrix is singular or nearly singular, "
                          "check for highly correlated features.")
        
        

    def predict(self, X):
        if self.beta is None:
            raise NotFittedError("This LinearRegression instance is not fitted yet. "
                                   "Call 'fit' with appropriate data before using this estimator.")

        if isinstance(X, pd.DataFrame):
            X = pd.get_dummies(X, drop_first=True)  # Drop first to avoid multicollinearity
        
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
    
            