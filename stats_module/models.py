import numpy as np
import pandas as pd
from stats_module.utils import *

class OLS:
    def __init__(self, include_intercept=True):
        """
        Ordinary Least Squares (OLS) regression model.

        Parameters
        ----------
        include_intercept : bool, default=True
            Whether to include an intercept in the model.
        """
        self.include_intercept = include_intercept
        self.beta = None
        self.modeltype = OLS

    def fit(self, X, y, use_gradient_descent=False, max_iter=1000, alpha=0.01, tol=1e-6):
        """
        Fit the OLS model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.
        use_gradient_descent : bool, default=False
            Whether to use gradient descent for optimization.
        max_iter : int, default=1000
            Maximum number of iterations for gradient descent.
        alpha : float, default=0.01
            Learning rate for gradient descent.
        tol : float, default=1e-6
            Tolerance for convergence in gradient descent.
        """
        # validate the input data
        validate_data(X, y)

        # add intercept if required
        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X

        if use_gradient_descent:
            # initialize parameters for gradient descent
            beta = np.zeros(X_.shape[1])
            cost = np.zeros(max_iter)

            for i in range(max_iter):
                y_hat = X_ @ beta
                residuals = y - y_hat
                gradient = -2 * X_.T @ residuals / len(y)
                beta -= alpha * gradient
                cost[i] = np.sum(residuals**2) / len(y)

                # check for convergence
                if i > 0 and abs(cost[i] - cost[i - 1]) < tol:
                    print(f"Converged after {i} iterations")
                    break

            if i == max_iter - 1:
                print("Warning: Gradient descent did not converge.")

            self.beta = beta
        else:
            # closed-form solution for OLS
            self.beta = np.linalg.inv(X_.T @ X_) @ X_.T @ y
            cond = np.linalg.cond(X_.T @ X_)
            if cond > 1e10:
                print("Warning: Matrix is ill-conditioned. Consider using regularization.")
    
    def predict(self, X):
        """
        Predict using the fitted OLS model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.

        Returns
        -------
        numpy array
            Predicted values.
        """
        if self.beta is None:
            raise ValueError("This OLS instance is not fitted yet. Call 'fit' first.")

        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X
        return X_ @ self.beta
    

    def estimate_variance(self, X, y):
        """
        Estimate the variance-covariance matrix of the coefficients.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.

        Returns
        -------
        numpy array
            Variance-covariance matrix.
        """
        y_hat = self.predict(X)
        sigma_hat = sigma_hat_corr(X, y, y_hat)
        return sigma_hat * np.linalg.inv(X.T @ X)
    

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
            Diagonal values of the hat matrix (leverages).
        """
        if self.beta is None:
            raise ValueError("This OLS instance is not fitted yet. Call 'fit' first.")

        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X
        h = X_ @ np.linalg.inv(X_.T @ X_) @ X_.T
        return np.diag(h)
 
    

class GLS:
    def __init__(self, include_intercept=True):
        """
        Generalized Least Squares (GLS) regression model.

        Parameters
        ----------
        include_intercept : bool, default=True
            Whether to include an intercept in the model.
        """
        self.include_intercept = include_intercept
        self.beta = None
        self.sigma = None
        self.modeltype = GLS

    def fit(self, X, y, sigma):
        """
        Fit the GLS model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.
        sigma : numpy array
            The covariance matrix of the errors.
        """
        # validate the input data
        validate_data(X, y)

        if isinstance(sigma, pd.DataFrame):
            sigma = sigma.values

        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X

        try:
            self.beta = np.linalg.inv(X_.T @ np.linalg.inv(sigma) @ X_) @ X_.T @ np.linalg.inv(sigma) @ y
        except np.linalg.LinAlgError:
            raise ValueError("Feature matrix is singular or nearly singular. Check for multicollinearity.")

        self.sigma = sigma

    def predict(self, X):
        """
        Predict using the fitted GLS model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.

        Returns
        -------
        numpy array
            Predicted values.
        """
        if self.beta is None:
            raise ValueError("This GLS instance is not fitted yet. Call 'fit' first.")

        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X
        return X_ @ self.beta

    
    def estimate_variance(self, X, y):
        """
        Estimate the variance-covariance matrix of the coefficients.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.

        Returns
        -------
        numpy array
            Variance-covariance matrix.
        """
        y_hat = self.predict(X)
        np.diag(y - y_hat)
        XTX = X.T @ X
        return 1/(len(y) - X.shape[1]) * np.linalg.inv(XTX) @ X.T @ np.diag(y - y_hat) @ X @ np.linalg.inv(XTX)
    
class Ridge:
    def __init__(self, alpha=1.0, include_intercept=True):
        """
        Ridge regression model.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength.
        include_intercept : bool, default=True
            Whether to include an intercept in the model.
        """
        self.alpha = alpha
        self.include_intercept = include_intercept
        self.beta = None
        self.modeltype = Ridge

    def fit(self, X, y):
        """
        Fit the Ridge regression model to the data.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.
        """
        # validate the input data
        validate_data(X, y)

        # add intercept if required
        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X

        # closed-form ridge regression solution
        n_features = X_.shape[1]
        I = np.eye(n_features)
        if self.include_intercept:
            I[0, 0] = 0  # don't regularize the intercept

        try:
            self.beta = np.linalg.inv(X_.T @ X_ + self.alpha * I) @ X_.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("Feature matrix is singular or nearly singular. Check for multicollinearity or increase regularization strength.")

    def predict(self, X):
        """
        Predict using the fitted Ridge model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.

        Returns
        -------
        numpy array
            Predicted values.
        """
        if self.beta is None:
            raise ValueError("This Ridge instance is not fitted yet. Call 'fit' first.")

        X_ = np.column_stack([np.ones(X.shape[0]), X]) if self.include_intercept else X
        return X_ @ self.beta

    def residuals(self, X, y):
        """
        Calculate residuals of the fitted Ridge model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.

        Returns
        -------
        numpy array
            Residuals of the model.
        """
        y_hat = self.predict(X)
        return y - y_hat


    def estimate_variance(self, X, y):
        """
        Estimate the variance-covariance matrix of the coefficients.

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            The feature matrix.
        y : numpy array or pandas Series
            The target vector.

        Returns
        -------
        numpy array
            Variance-covariance matrix.
        """
        y_hat = self.predict(X)
        XTX = X.T @ X
        reg_term = self.alpha * np.eye(XTX.shape[0])
        inv_term = np.linalg.inv(XTX + reg_term)
        var_est = sigma_hat_corr(X, y, y_hat) * inv_term @ XTX @ inv_term
        print("Warning: Variance estimator assumes homoskedasticity.")
        return var_est
    
class ReducedModel:
    def __init__(self, base_model, selected_features):
        """
        Wrapper class for reduced models.

        Parameters
        ----------
        base_model : object
            The base model instance (e.g., OLS, Ridge, etc.).
        selected_features : list
            List of selected feature indices for the reduced model (e.g, [0,1,2]).
        """
        if base_model.beta is None:
            raise ValueError("The base model must be fitted before creating a ReducedModel.")

        self.base_model = base_model
        self.selected_features = selected_features
        self.beta_reduced = self._extract_reduced_beta()

    def _extract_reduced_beta(self):
        """
        Extract coefficients corresponding to the selected features.

        Returns
        -------
        numpy array
            Coefficients for the reduced model, including intercept if applicable.
        """
        if self.base_model.include_intercept:
            beta_reduced = np.zeros(len(self.selected_features) + 1)
            beta_reduced[0] = self.base_model.beta[0]  # Intercept
            for i, feature_idx in enumerate(self.selected_features):
                beta_reduced[i + 1] = self.base_model.beta[feature_idx + 1]
            return beta_reduced
        else:
            return self.base_model.beta[self.selected_features]
        
    @property
    def beta(self):
        """
        Return the reduced model coefficients.

        Returns
        -------
        numpy array
            The reduced coefficients.
        """
        return self.beta_reduced

    def predict(self, X):
        """
        Predict using the reduced set of features.

        Parameters
        ----------
        X : numpy array
            The full feature matrix.

        Returns
        -------
        numpy array
            Predicted values using the reduced model coefficients.
        """
        if self.base_model.include_intercept:
            X_reduced = np.column_stack([np.ones(X.shape[0]), X[:, self.selected_features]])
        else:
            X_reduced = X[:, self.selected_features]
        return X_reduced @ self.beta_reduced
    


def summary(model, X, y):
    """
    Generate a summary of the model's performance.

    Parameters
    ----------
    model : object
        The regression model (OLS, GLS, or Ridge). Must have a `predict` method.
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix of shape `(n_samples, n_features)`.
    y : numpy.ndarray or pandas.Series
        Response vector of length `n_samples`.

    Returns
    -------
    dict
        A dictionary containing:
        - coefficients : numpy.ndarray
            The fitted coefficients of the model.
        - r_squared : float
            The R-squared value, representing the proportion of variance explained by the model.
    """
    if not hasattr(model, "predict") or not hasattr(model, "beta"):
        raise ValueError("The provided model must have 'predict' and 'beta' attributes.")

    y_hat = model.predict(X)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_hat)**2)
    r_squared = 1 - ss_res / ss_total

    return {'coefficients': model.beta, 'r_squared': r_squared}