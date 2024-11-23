import numpy as np
import pandas as pd
from sklearn import NotFittedError
from sklearn.preprocessing import OneHotEncoder

class OLS:
    def __init__(self, include_intercept=True):
        self.include_intercept = include_intercept
        self.beta = None

    #should we make use_gradient_descent a parameter automatically use it if p>threshold?
    def fit(self, X, y):
        if y.ndim > 1:
            raise ValueError("y must be a 1-dimensional array.")
        
        #raise error if X and y have different number of observations
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")
        
        #raise error if X has categorical variables
        if isinstance(X, pd.DataFrame) and X.select_dtypes(include='object').shape[1] > 0:
            raise ValueError("X cannot have categorical variables.")
            
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
            raise NotFittedError("This OLS instance is not fitted yet. "
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
    
            