import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import sklearn.linear_model as lm
from dtuimldmtools import rlr_validate


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

    # Calculate the baseline error for the training set
    Error_train_baseline = np.sum(y_train != most_common_class) / len(y_train)

    # Calculate the baseline error for the test set
    pred = y_test != most_common_class
    Error_test_baseline = np.sum(pred) / len(y_test)

    # Return the baseline errors
    return Error_test_baseline, Error_train_baseline, pred.astype(int)


def train(
    train_set,
    verbose,
    alphas,
    K_inner=10,
):
    """
    Trains a logistic regression model using the given training set.

    Parameters:
    train_set (tuple): A tuple containing the training data X_train and the corresponding labels y_train.
    verbose (bool): If True, prints progress messages during training. If False, training is silent.
    alphas (list): A list of regularization parameters to be used in the logistic regression model.
    K_inner (int, optional): The number of folds to use for inner cross-validation. Defaults to 10.

    Returns:
    model: The trained logistic regression model.
    y_train_est: The predicted labels for the training data.
    """

    # Split the data into X and y and convert to numpy arrays
    X_train, y_train = train_set

    # Standardize the features
    standardize = StandardScaler()
    X_train = standardize.fit_transform(X_train)

    # Create a logistic regression model with cross-validation
    model = lm.LogisticRegressionCV(
        penalty="l2", Cs=1 / alphas, cv=K_inner, max_iter=1000
    )

    # Fit the best model from the inner cv to the training data
    model = model.fit(X_train, y_train)

    # Predict the labels for the training data
    y_train_est = model.predict(X_train)

    # Return the trained model and the predicted labels
    return model, y_train_est


def error_rate(model, test_set):
    """
    Calculate the error rate of a given model on a test set.

    Parameters:
    model (object): The trained model to evaluate.
    test_set (tuple): A tuple containing the test features (X_test) and the corresponding labels (y_test).

    Returns:
    float: The error rate, defined as the proportion of misclassified samples in the test set.
    predictions (array): An array of 0s and 1s indicating whether each sample was misclassified (1) or not (0).
    """

    # Extract the test features and labels from the test set tuple
    X_test, y_test = test_set

    # Predict the labels for the test features using the best model from the inner cv
    y_test_est = model.predict(X_test)

    # Calculate the error rate by counting the number of misclassified samples and dividing by the total number of samples
    pred = y_test_est != y_test

    test_err = np.sum(pred) / len(y_test)

    return test_err, pred.astype(int)


def train_eval(train_inner, train_outer, test_set, alphas, K_inner=10, verbose=0):
    """
    Trains a logistic regression model on the given train_set and evaluates its performance on the test_set.

    Parameters:
        train_set (array-like): The training dataset.
        test_set (array-like): The test dataset.
        alphas (list): List of regularization parameters (alphas).
        K_inner (int, optional): The number of folds for inner cross-validation. Defaults to 10.
        verbose (int, optional): Verbosity level. Set to 0 for no output during training. Defaults to 0.

    Returns:
        tuple: A tuple containing the test error rate, the index of the optimal alpha, the optimal alpha value, and the trained model.
        predictions (array): An array of 0s and 1s indicating whether each sample was misclassified (1) or not (0) on test.
    """
    # Train the logistic regression model on the training set
    model, y_train_est = train(train_inner, verbose, alphas, K_inner)
    # Find the index of the optimal alpha value in the list of alphas
    opt_alpha_idx = np.where(1 / np.array(alphas) == model.C_[0])[0][0]
    # Get the optimal alpha value corresponding to the index
    opt_alpha = alphas[opt_alpha_idx]

    # Train and test for outer loop
    model = lm.LogisticRegression(C=opt_alpha, max_iter=1000, penalty="l2")
    # 
    model = model.fit(train_outer[0], train_outer[1])
    # Calculate the error rate of the trained model on the test set
    
    train_err, _ = error_rate(model, train_outer)
    test_err, pred = error_rate(model, test_set)

    # Return the test error rate, optimal alpha index, optimal alpha value, and the trained model
    return test_err, train_err, opt_alpha_idx, opt_alpha, model, pred
