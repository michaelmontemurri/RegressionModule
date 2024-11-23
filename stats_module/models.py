import numpy as np
import pandas as pd
from stats_module.utils import *


class OLS:
    """
    Ordinary Least Squares (OLS) regression model.

    Attributes
    ----------
    include_intercept : bool
        Indicates whether to include an intercept term in the model.
    beta : numpy array
        The estimated coefficients of the regression model.
    """
    def __init__(self, include_intercept=True):
        """
        Initialize the OLS model.

        Parameters
        ----------
        include_intercept : bool, optional
            Whether to include an intercept term in the model (default is True).
        """
        self.include_intercept = include_intercept
        self.beta = None

    def gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000, tolerance=1e-6, return_loss=False):
        """
        Perform gradient descent to fit the OLS model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix (independent variables).
        y : numpy array or pandas Series
            The response variable (dependent variable).
        learning_rate : float, optional
            The learning rate for gradient descent (default is 0.01).
        n_iterations : int, optional
            The number of iterations for gradient descent (default is 1000).

        Returns
        -------
        None
        """
            # Initialize coefficients and storage for loss
        self.beta = np.zeros(X.shape[1])
        loss_history = []

        for _ in range(n_iterations):
            residuals = y - X @ self.beta
            gradient = -2 / X.shape[0] * X.T @ residuals
            beta_new = self.beta - learning_rate * gradient

            # Record loss
            loss = np.mean(residuals ** 2)
            loss_history.append(loss)

            # Check for convergence
            if np.allclose(beta_new, self.beta, atol=tolerance):
                self.beta = beta_new
                break

            self.beta = beta_new

        if return_loss:
            return loss_history

    def fit(self, X, y, use_gradient_descent=False):
        """
        Fit the OLS model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix (independent variables).
        y : numpy array or pandas Series
            The response variable (dependent variable).
        use_gradient_descent : bool, optional
            Whether to use gradient descent for fitting (default is False).

        Returns
        -------
        None
        """
        # Validate input data
        validate_data(X, y)

        # Add intercept term if required
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        if use_gradient_descent:
            # Logic for gradient descent (to be implemented later)
            self.gradient_descent(X_, y)
            pass
        else:
            # Fit using the normal equation
            try:
                self.beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y
            except np.linalg.LinAlgError:
                raise Warning(
                    "Feature matrix is singular or nearly singular. "
                    "Check for highly correlated features."
                )

    def predict(self, X):
        """
        Predict using the fitted OLS model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix for prediction.

        Returns
        -------
        numpy array
            Predicted values for the given feature matrix.
        """
        if self.beta is None:
            raise ValueError(
                "This OLS instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this estimator."
            )
        
        # Add intercept term if required
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        return X_ @ self.beta

    def estimate_variance(self, X, y):
        """
        Estimate the variance of the residuals.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The response variable.

        Returns
        -------
        float
            Estimated variance of the residuals.
        """
        y_hat = self.predict(X)
        return sigma_hat_corr(X, y, y_hat)

    def leverages(self, X):
        """
        Calculate the leverage values for each observation.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.

        Returns
        -------
        numpy array
            Leverage values for each observation.
        """
        if self.beta is None:
            raise ValueError(
                "This OLS instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this estimator."
            )
        
        # Add intercept term if required
        if self.include_intercept:
            X_ = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_ = X

        h = X_ @ np.linalg.inv(X_.T @ X_) @ X_.T
        return np.diag(h)

    def residuals(self, X, y):
        """
        Calculate the residuals (differences between observed and predicted values).

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The response variable.

        Returns
        -------
        numpy array
            Residuals for the given data.
        """
        y_hat = self.predict(X)
        return y - y_hat

    def summary(self, X, y):
        """
        Generate a summary of the fitted model, including coefficients and R-squared.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The response variable.

        Returns
        -------
        dict
            Summary containing coefficients and R-squared value.
        """
        y_hat = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
        ss_res = np.sum((y - y_hat)**2)         # Residual sum of squares
        r_squared = 1 - ss_res/ss_total
        #regularized r_squared
        #r_squared = 1 - (ss_res/(X.shape[0]-X.shape[1]-1))/(ss_total/(X.shape[0]-1))
        return {'coefficients': self.beta, 'r_squared': r_squared}


    
            