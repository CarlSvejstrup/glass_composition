import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import sklearn.linear_model as lm


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import numpy as np


def baseline(train_set, test_set):
    """
    Calculate the baseline error for a classification problem.

    Parameters:
    train_set (tuple): A tuple containing the training data and labels.
    test_set (tuple): A tuple containing the test data and labels.

    Returns:
    float: The baseline error for the test set.
    float: The baseline error for the training set.
    predictions (array): An array of 0s and 1s indicating whether each sample was misclassified (1) or not (0).
    """

    # Initialize variables for baseline error
    Error_test_baseline = None
    Error_train_baseline = None

    # Extract labels from train_set and test_set
    _, y_train = train_set
    _, y_test = test_set

    # Find the most common class in the training set
    most_common_class = np.bincount(y_train).argmax()
    # Predict the most common class for all samples
    pred = np.full(len(y_test), most_common_class)

    # Calculate the baseline error for the training set
    Error_train_baseline = np.sum(y_train != most_common_class) / len(y_train)

    # Calculate the baseline error for the test set
    Error_test_baseline = np.sum(y_test != most_common_class) / len(y_test)

    # Return the baseline errors
    return Error_test_baseline, Error_train_baseline, pred


def train(X_train, y_train, alphas, K_inner=10):
    """
    Modified training function with manual cross-validation to compute
    errors for each alpha.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        alphas (list): List of alphas for regularization.
        K_inner (int): Number of folds for cross-validation.

    Returns:
        tuple: Best model, training errors, validation errors, best alpha.
    """

    kf = KFold(n_splits=K_inner)
    train_errors = np.zeros(len(alphas))
    val_errors = np.zeros(len(alphas))
    weights = np.zeros((len(alphas), X_train.shape[1] + 1))  # +1 for bias term

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Standardize features within the fold
        scaler_X = StandardScaler()
        X_train_fold = scaler_X.fit_transform(X_train_fold)
        X_val_fold = scaler_X.transform(X_val_fold)

        X_val_fold = np.concatenate((np.ones((X_val_fold.shape[0], 1)), X_val_fold), 1)
        X_train_fold = np.concatenate(
            (np.ones((X_train_fold.shape[0], 1)), X_train_fold), 1
        )

        for i, alpha in enumerate(alphas):
            model = LogisticRegression(penalty="l2", C=1 / alpha, max_iter=1000)
            model.fit(X_train_fold, y_train_fold)

            # Predict and calculate errors
            y_train_pred = model.predict(X_train_fold)
            y_val_pred = model.predict(X_val_fold)

            # Calculate error rates
            train_error = np.mean(y_train_pred != y_train_fold)
            val_error = np.mean(y_val_pred != y_val_fold)

            train_errors[i] += train_error
            val_errors[i] += val_error

            # Save the weights
            weights[i] = model.coef_.flatten()
            # Add the bias term to the weight vector
            weights[i, 0] = model.intercept_

    # Select best alpha based on validation error
    best_alpha_idx = np.argmin(val_errors)
    best_alpha = alphas[best_alpha_idx]

    # Average the errors over the folds
    train_errors /= K_inner
    val_errors /= K_inner

    return train_errors, val_errors, best_alpha, weights


def train_eval(train_inner, train_outer, test_outer, alphas, K_inner=10):
    """
    Trains and evaluates the logistic regression model.

    Parameters:
        train_inner (tuple): Training dataset for model selection.
        test_outer (tuple): Test dataset for final evaluation.
        alphas (list): List of alphas for regularization.
        K_inner (int): Number of folds for cross-validation.

    Returns:
        tuple: Test error, training errors, validation errors, best alpha, final model.
    """
    X_train_inner, y_train_inner = train_inner
    X_test_outer, y_test_outer = test_outer

    # Train the model and get the best model and errors
    train_err_vs_lambda, val_err_vs_lambda, best_alpha, weights = train(
        X_train_inner, y_train_inner, alphas, K_inner
    )

    # Train and test for outer loop
    model = lm.LogisticRegression(C=1 / best_alpha, max_iter=1000, penalty="l2")
    #
    X_train_outer, y_train_outer = train_outer

    # Adding bias term
    X_train_outer = np.concatenate(
        (np.ones((X_train_outer.shape[0], 1)), X_train_outer), 1
    )

    X_test_outer = np.concatenate(
        (np.ones((X_test_outer.shape[0], 1)), X_test_outer), 1
    )

    model = model.fit(X_train_outer, y_train_outer)

    # Train error rate
    y_train_pred = model.predict(X_train_outer)
    train_error = np.mean(y_train_pred != y_train_outer)

    # Test error rate
    y_test_pred = model.predict(X_test_outer)
    test_error = np.mean(y_test_pred != y_test_outer)

    return (
        test_error,
        train_error,
        train_err_vs_lambda,
        val_err_vs_lambda,
        best_alpha,
        model,
        y_test_pred,
        weights,
    )
