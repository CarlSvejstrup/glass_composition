import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# ANN
import torch
from torch import nn, utils
from torch.utils.data import DataLoader, Dataset, Subset


from sklearn import model_selection
import sklearn.linear_model as lm
from dtuimldmtools import rlr_validate

import nn_classifier as nn_class
import nn_regressor as nn_reg
import linear_regression_regularized as rlr
from data_loader import *
import logistic_regression as log_reg
import statistical_tests as stats


# TODO
# - Seperate the standardization in inner and outer loop
# - Train eval can be for inner or outer loop (nn)

np.random.seed(13)
torch.manual_seed(13)


def outer_layer(
    X,
    y,
    batch_size,
    hidden_neurons_list,
    alphas,
    plot,
    epochs=10,
    K_outer=10,
    K_inner=5,
    verbose=1,
    nn_standardize=False,
    regression=True,
):
    """
    Perform outer layer of k-fold cross-validation for regression or classification tasks.

    Parameters:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Target values.
    - batch_size (int): Number of samples per batch for neural network training.
    - hidden_neurons_list (list): List of integers representing the number of hidden neurons to evaluate.
    - alphas (list): List of regularization parameters for regularized linear regression.
    - plot (bool): Whether to plot regularization results for the last fold.
    - epochs (int): Number of epochs for neural network training. Default is 10.
    - K_outer (int): Number of outer folds for k-fold cross-validation. Default is 10.
    - K_inner (int): Number of inner folds for k-fold cross-validation. Default is 5.
    - verbose (int): Verbosity level. Set to 0 for no output, 1 for progress updates, and 2 for detailed output. Default is 1.
    - nn_standardize (bool): Whether to standardize the data for the neural network. Default is False.
    - regression (bool): Whether the task is regression or classification. Default is True.


    Returns:
    - If regression is True:
        - inner_fold_optimal_hidden_neurons (int): Number of hidden neurons with the lowest error.
        - test_errors (numpy.ndarray): Test errors for each outer fold.
        - hidden_neurons_best_error (float): Lowest error achieved among all hidden neuron configurations.
    - If regression is False:
        - Error_test_regression (numpy.ndarray): Test errors for regularized linear regression.
        - Error_test_baseline (numpy.ndarray): Test errors for baseline model.
        - test_errors (numpy.ndarray): Test errors for neural network classification.

    """

    # Store errors for regression and baseline for each fold
    Error_train_regression = np.empty((K_outer, 1))
    Error_test_regression = np.empty((K_outer, 1))
    Error_train_baseline = np.empty((K_outer, 1))
    Error_test_baseline = np.empty((K_outer, 1))
    w_rlr_arr = []

    # Store test errors for NN for each outer fold
    test_errors = np.empty((K_outer, 1))
    overall_best_hidden_neurons = None
    overall_lowest_error_nn = float("inf")

    kf_outer = model_selection.KFold(n_splits=K_outer, shuffle=True)

    if regression:
        print("Running regression")

        # outer k-fold loop
        for i, (train_idx, test_idx) in enumerate(kf_outer.split(X, y)):

            # Print progress
            if verbose >= 0:
                print(f"\nStarting Outer Fold {i+1}/{K_outer}")
                print("-" * 30)

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            # Training data for inner loop
            train_data = (X_train, y_train)

            # Standardize the features based on the training set
            if nn_standardize:
                scaler = StandardScaler()
                # Standardize the features based on the training set

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                outer_train_data = (X_train, y_train)
                outer_test_data = (X_test, y_test)

            else:
                outer_train_data = (X_train, y_train)
                outer_test_data = (X_test, y_test)

            # Regalurized linear regression
            (
                opt_lambda,
                opt_val_err,
                mean_w_vs_lambda,
                train_err_vs_lambda,
                test_err_vs_lambda,
            ) = rlr.train_rlr(train_data, alphas=alphas, K_inner=K_inner)

            # Testing regression for the current fold
            Error_test_regression[i], w_rlr, squared_err_rlr = rlr.test_rlr(
                outer_test_data, outer_train_data, opt_lambda
            )
            w_rlr_arr.append(w_rlr)

            # Baseline testing for the current fold
            Error_test_baseline[i], Error_train_baseline[i], squared_err_baseline = (
                rlr.lr_baseline(train_data, outer_test_data)
            )

            # Train and evaluate hidden neurons from the inner layers
            (
                inner_fold_optimal_hidden_neurons,
                inner_fold_best_model,
                inner_fold_lowest_error,
                hidden_neurons_best_error,
            ) = nn_reg.nested_layer(
                dataset=train_data,
                hidden_neurons=hidden_neurons_list,
                batch_size=batch_size,
                standardize=nn_standardize,
                epochs=epochs,
                verbose=verbose,
                K_inner=K_inner,
            )
            test_error_nn, _, _, squared_err_nn = nn_reg.outer_test(
                train_set=outer_train_data,
                test_set=outer_test_data,
                hidden_neuron=inner_fold_optimal_hidden_neurons,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            test_errors[i] = test_error_nn

            # Print progress
            if verbose >= 0:
                print(
                    f"{'Baseline':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {Error_test_baseline[i][0]:.2e}"
                )
                print(
                    f"{'rlr':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {Error_test_regression[i][0]:.2e} with {opt_lambda:.2e} Lambda"
                )

                print(
                    f"{'ANN regression':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {test_error_nn:.2e} with {inner_fold_optimal_hidden_neurons} Neurons"
                )

                print("---" * 20)

            stats.t_test(
                squared_err_rlr=squared_err_rlr,
                squared_err_nn=squared_err_nn,
                squared_err_baseline=squared_err_baseline,
            )

            # Plot different regeralization rates for the last fold
            if plot and i == K_outer - 1:
                rlr.plot_rlr(
                    alphas,
                    alphas,
                    mean_w_vs_lambda,
                    train_err_vs_lambda,
                    test_err_vs_lambda,
                    opt_lambda,
                    index=0,
                )

        return (
            inner_fold_optimal_hidden_neurons,
            test_errors,
            hidden_neurons_best_error,
        )

    if regression == False:
        print("Running classification")

        # binary classification
        y = np.where(y > np.median(y), 1, 0)

        # outer k-fold loop
        for i, (train_idx, test_idx) in enumerate(kf_outer.split(X, y)):
            # Print progress
            if verbose >= 0:
                print(f"\nStarting Outer Fold {i+1}/{K_outer}")
                print("-" * 30)

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            # Training data for inner loop
            train_data = (X_train, y_train)

            # Standardize the features based on the training set
            if nn_standardize:
                scaler = StandardScaler()
                # Standardize the features based on the training set

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                outer_train_data = (X_train, y_train)
                outer_test_data = (X_test, y_test)

            else:
                outer_train_data = (X_train, y_train)
                outer_test_data = (X_test, y_test)

            # Baseline testing for the current fold for classification for the dominant class
            Error_test_baseline[i], Error_train_baseline[i], prediction_baseline = (
                log_reg.baseline(train_data, outer_test_data)
            )

            # Train and evaluate hidden neurons
            (
                inner_fold_optimal_hidden_neurons,
                inner_fold_best_model,
                inner_fold_lowest_error,
                hidden_neurons_best_error,
            ) = nn_class.nested_layer(
                dataset=train_data,
                hidden_neurons=hidden_neurons_list,
                batch_size=batch_size,
                standardize=nn_standardize,
                epochs=epochs,
                verbose=verbose,
                K_inner=K_inner,
            )

            error_rate_nn, prediction_nn = nn_class.outer_test(
                outer_train_data,
                outer_test_data,
                hidden_neuron=inner_fold_optimal_hidden_neurons,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            test_errors[i] = error_rate_nn

            # Testing regression for the current fold
            test_err_log_reg, opt_alpha_idx, opt_alpha, model, prediction_log_reg = (
                log_reg.train_eval(
                    train_data,
                    outer_train_data,
                    outer_test_data,
                    alphas,
                    K_inner,
                    verbose=verbose,
                )
            )

            # Print progress
            if verbose >= 0:
                print(
                    f"{'Baseline':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {Error_test_baseline[i][0]:.3e}"
                )
                print(
                    f"{'Logistic regression':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {test_err_log_reg:.3e} with {opt_alpha:.2e} Lambda"
                )

                print(
                    f"{'ANN classification':<20}: Outer Fold {i+1}/{K_outer}: Test Error: {error_rate_nn:.3e} with {inner_fold_optimal_hidden_neurons} Neurons"
                )
                print("---" * 20)

            stats.mc_nemar(
                y_true=y_test,
                pred_nn=prediction_nn,
                pred_baseline=prediction_baseline,
                pred_log_reg=prediction_log_reg,
                alpha=0.05,
            )

        return Error_test_regression, Error_test_baseline, test_errors

    # Importing the data


y = np.asanyarray((df["RI"]).squeeze())
X = np.asanyarray(df.drop(["RI"], axis=1))

N, M = X.shape

neurons = [1, 2, 3, 5, 7, 10, 15, 20]
alphas = np.logspace(-5, 9, num=40)


# 0 - Silent, no output.
# 1 - Basic progress information and critical results.
# 2 - Detailed progress, including epoch-wise or step-wise updates.
# 3 - Very detailed output, including batch-level updates, gradients, or more in-depth debugging information.

outer_layer(
    X,
    y,
    batch_size=5,
    hidden_neurons_list=neurons,
    epochs=25,
    K_outer=10,
    K_inner=10,
    verbose=0,
    alphas=alphas,
    plot=True,
    nn_standardize=False,
    regression=True,
)
