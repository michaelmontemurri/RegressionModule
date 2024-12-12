# Regression Module

A package that implements estimators, hypothesis testing, etc. from scratch. We provide detailed examples and usage in the accompanying jupyter notebook `Documentation.ipnyb`.

---

## **Project Structure**
```
stats_module_project/

├── notebooks/             # notebooks with implementation examples and documentation

├── stats_module/          # package for methods
│   ├── __init__.py        # makes the directory a Python package
│   ├── models.py          # all implemented model classes
│   ├── loss_estimation.py # loss estimation methods
│   ├── testing.py         # testing methods
│   ├── utils.py           # utility function

├── venv/                  # virtual environment (excluded in .gitignore)
├── .gitignore             # stuff to ignore in the repository
├── README.md              # project overview and usage instructions

├── requirements.txt       # dependencies
├── setup.py               # packaging configuration
```
---

## **Features**

The `stats_module` provides the following functionalities:

- **`models.py`**:
  - Implements the OLS, GLS, and Ridge estimators with accompanying functionalities
 
- **`loss_estimation.py`**:
  - Implements the naive loss estimator and training testing loss estimator for general models. Implements the leave-one-out estimator for linear models. 

- **`testing.py`**:
  - Implements basic hypothesis tests such as t-test and F-test, along with confidence intervals.
  
- **`utils.py`**:
  - Implements basic utility functions which are used commonly across the project.

Each module is demonstrated in **Documentation** for clarity and ease of use.
