import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import sklearn.linear_model as lm
from dtuimldmtools import rlr_validate


def lr_baseline(train_set, test_set):
    """
    Calculate the baseline error for linear regression.

    Parameters:
    train_set (tuple): A tuple containing the training data (X_train, y_train).
    test_set (tuple): A tuple containing the test data (X_test, y_test).

    Returns:
    tuple: A tuple containing the baseline test error, baseline train error, and squared errors.
    """

    # Initialize variables
    Error_test_baseline = None
    Error_train_baseline = None

    # Extract labels from train_set and test_set
    _, y_train = train_set
    _, y_test = test_set

    # Calculate the baseline error for the training set
    Error_train_baseline = np.mean(np.square(y_train - np.mean(y_train)))

    # Calculate the baseline error for the test set
    squared_errors = np.square(y_test - np.mean(y_test))
    Error_test_baseline = np.mean(squared_errors)

    return Error_test_baseline, Error_train_baseline, squared_errors


def train_rlr(train_set, alphas, K_inner=5):
    """
    Trains a regularized linear regression model using the given training set.

    Parameters:
    train_set (tuple): A tuple containing the training data X_train and the corresponding labels y_train.
    alphas (list): A list of regularization parameters to be tested.
    K_inner (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
    tuple: A tuple containing the optimal regularization parameter (opt_lambda), the optimal validation error (opt_val_err),
    the mean weights versus regularization parameter (mean_w_vs_lambda), the training error versus regularization parameter (train_err_vs_lambda),
    and the test error versus regularization parameter (test_err_vs_lambda).
    """
    # Split the data into X and y and convert to numpy arrays
    X_train, y_train = train_set

    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)

    # Run the rlr_validate function
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas=alphas, cvf=K_inner)

    return (
        opt_lambda,
        opt_val_err,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    )


def test_rlr(test_set, train_set, opt_lambda):
    """
    Perform regularized linear regression on the test set.

    Parameters:
    test_set (tuple): A tuple containing the test data features (X_test) and labels (y_test).
    train_set (tuple): A tuple containing the training data features (X_train) and labels (y_train).
    opt_lambda (float): The optimal lambda value for regularization.

    Returns:
    Error_test_rlr (float): The mean squared error of the predictions on the test set.
    w_rlr (ndarray): The coefficients of the regularized linear regression model.
    squared_errors (ndarray): The squared errors between the true labels and the predicted labels.
    """

    # Initialize variables
    w_rlr = None
    Error_test_rlr = None

    # Split the data into X and y and convert to numpy arrays
    X_train, y_train = train_set
    X_test, y_test = test_set

    # Add bias term to the data
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)

    # Regularized linear regression model
    rlr_model = lm.Ridge(alpha=opt_lambda, fit_intercept=True).fit(X_train, y_train)

    # Predict the labels 
    pred = rlr_model.predict(X_test)
    squared_errors = np.square(y_test - pred)
    # Calculate the mean squared error
    Error_test_rlr = np.square(y_test - pred).mean()

    # Weight vector w_rlr
    w_rlr = rlr_model.coef_
    # Add the bias term to the weight vector
    w_rlr[0] = rlr_model.intercept_

    return Error_test_rlr, w_rlr, squared_errors


def plot_rlr(
    alphas,
    lambdas,
    mean_w_vs_lambda,
    train_err_vs_lambda,
    test_err_vs_lambda,
    opt_lambda,
    index=0,
):
    """
    Plot the results of Ridge Regression with different regularization factors.

    Parameters:
    alphas (array-like): Array of regularization factors.
    lambdas (array-like): Array of regularization factors.
    mean_w_vs_lambda (array-like): Array of mean coefficient values for each lambda.
    train_err_vs_lambda (array-like): Array of squared errors for training data for each lambda.
    test_err_vs_lambda (array-like): Array of squared errors for test data for each lambda.
    opt_lambda (float): Optimal lambda value.
    index (int, optional): Index of the figure. Defaults to 0.

    Returns:
    None
    """
    plt.figure(index, figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
    plt.xlabel("Regularization factor")
    plt.ylabel("Mean Coefficient Values")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
    plt.loglog(
        alphas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
    )
    plt.xlabel("Regularization factor")
    plt.ylabel("Squared error (crossvalidation)")
    plt.legend(["Train error", "Validation error"])
    plt.grid()
