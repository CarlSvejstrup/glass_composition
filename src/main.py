import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# ANN
import torch
from sklearn import model_selection
import matplotlib.pyplot as plt

import nn_classifier as nn_class
import nn_regressor as nn_reg
import linear_regression_regularized as rlr
from data_loader import *
import logistic_regression as log_reg
import statistical_tests as stats


# TODO
# - Add more comments

np.random.seed(13)
torch.manual_seed(13)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


# Logging the results for test and training
def print_fold_results(
    i,
    K_outer,
    train_errors_baseline,
    train_errors_regression,
    train_errors_nn,
    test_errors_baseline,
    test_errors_regression,
    test_errors_nn,
):
    logger.info(f"\n==== Running Outer Fold {i+1}/{K_outer} ====")

    # Training Results
    logger.info("\nTraining Results:")
    logger.info("-" * 50)
    logger.info(f"{'Model':<20} | {'Train Error':<20} | {'Notes'}")
    logger.info(f"{'Baseline':<20} | {train_errors_baseline[i][0]:<20.2e} |")
    logger.info(
        f"{'RLR':<20} | {train_errors_regression[i][0]:<20.2e} | Lambda: {train_errors_regression[i][1]:.2e}"
    )
    logger.info(
        f"{'ANN Regression':<20} | {train_errors_nn[i][0]:<20.2e} | Neurons: {train_errors_nn[i][1]}"
    )

    # Testing Results
    logger.info("\nTesting Results:")
    logger.info("-" * 50)
    logger.info(f"{'Model':<20} | {'Test Error':<20} | {'Notes'}")
    logger.info("-" * 50)
    logger.info(f"{'Baseline':<20} | {test_errors_baseline[i][0]:<20.2e} |")
    logger.info(
        f"{'RLR':<20} | {test_errors_regression[i][0]:<20.2e} | Lambda: {test_errors_regression[i][1]:.2e}"
    )
    logger.info(
        f"{'ANN Regression':<20} | {test_errors_nn[i][0]:<20.2e} | Neurons: {test_errors_nn[i][1]}"
    )
    logger.info("=" * 50)


# Logging the results for the statistical tests
def print_statistical_test_results(results, t_test=True):
    # For T-Test (Regression)
    if t_test:
        logger.info("\n=== Statistical T-Test Results ===")
        logger.info(
            f"{'Model':<15} | {'Lower CI':<15} | {'Upper CI':<15} | {'p_value'}"
        )
        logger.info("-" * 60)
        logger.info(
            f"{'RLR':<15} | {results['rlr'][0]:<15.2e} | {results['rlr'][1]:<15.2e} |"
        )
        logger.info(
            f"{'NN':<15} | {results['nn'][0]:<15.2e} | {results['nn'][1]:<15.2e} |"
        )
        logger.info(
            f"{'Baseline':<15} | {results['baseline'][0]:<15.2e} | {results['baseline'][1]:<15.2e} |"
        )
        logger.info(
            f"{'RLR-Baseline':<15} | {results['rlr_baseline'][0][0]:<15.2e} | {results['rlr_baseline'][0][1]:<15.2e} | p-value: {results['rlr_baseline'][1]:.2e}"
        )
        logger.info(
            f"{'RLR-NN':<15} | {results['rlr_nn'][0][0]:<15.2e} | {results['rlr_nn'][0][1]:<15.2e} | p-value: {results['rlr_nn'][1]:.2e}"
        )
        logger.info(
            f"{'NN-Baseline':<15} | {results['nn_baseline'][0][0]:<15.2e} | {results['nn_baseline'][0][1]:<15.2e} | p-value: {results['nn_baseline'][1]:.2e}"
        )
        logger.info("-" * 60)

    # For McNemar's Test (Classification)
    else:
        logger.info("\n=== McNemar's Test Results ===")
        logger.info(
            f"{'Comparison':<20} | {'Statistic':<10} | {'CI Lower':<15} | {'CI Upper':<15} | {'p-value':<10}"
        )
        logger.info("-" * 60)
        logger.info(
            f"{'NN vs. Baseline':<20} | {results['nn_baseline'][0]:<10.2f} | {results['nn_baseline'][1][0]:<15.2e} | {results['nn_baseline'][1][1]:<15.2e} | {results['nn_baseline'][2]:<10.2e}"
        )
        logger.info(
            f"{'NN vs. LogReg':<20} | {results['nn_logreg'][0]:<10.2f} | {results['nn_logreg'][1][0]:<15.2e} | {results['nn_logreg'][1][1]:<15.2e} | {results['nn_logreg'][2]:<10.2e}"
        )
        logger.info(
            f"{'LogReg vs. Baseline':<20} | {results['baseline_logreg'][0]:<10.2f} | {results['baseline_logreg'][1][0]:<15.2e} | {results['baseline_logreg'][1][1]:<15.2e} | {results['baseline_logreg'][2]:<10.2e}"
        )
        # Assuming similar return structure for other comparisons, repeat the logging format for them
        logger.info("-" * 60)


# Function for plotting the results for the learning curves
def plot_learning_curves(learning_curves):
    # Create a square figure with equal width and height, e.g., 6x6 inches.
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, curve in enumerate(learning_curves):
        ax.plot(curve, label=f"Fold {i+1}")
    ax.set_title("Learning Curves for Neural Network")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error")
    ax.set_ylim([0, 1])
    # Removed the ax.set_aspect("equal") for maintaining the figure's square appearance without forcing axis scaling.
    ax.legend()
    plt.show()


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
        - Error_train_regression (numpy.ndarray): Tuple of test errors and optimal alpha for linear regression, for each fold.
        - Error_test_regression (numpy.ndarray): Tuple of test errors and optimal alpha for linear regression, for each fold.
        - Error_train_baseline (numpy.ndarray): Training errors for baseline model.
        - Error_test_baseline (numpy.ndarray): Test errors for baseline model.
        - Error_train_nn (numpy.ndarray): Tuple of lowest training error and optimal hidden neurons for neural network regressor, for each fold
        - Error_test_nn (numpy.ndarray): Tuple of lowest test errors and optimal hidden neurons for neural network regressor, for each fold
        - learning_curves (numpy.ndarray): Training errors for neural network regressor.

    - If regression is False:
        - Error_train_regression (numpy.ndarray): Tuple of test errors and optimal alpha for logistic regression, for each fold.
        - Error_test_regression (numpy.ndarray): Tuple of test errors and optimal alpha for logistic regression, for each fold.
        - Error_train_baseline (numpy.ndarray): Training errors for baseline model.
        - Error_test_baseline (numpy.ndarray): Test errors for baseline model.
        - Error_train_nn (numpy.ndarray): Tuple of lowest training error and optimal hidden neurons for neural network classification, for each fold
        - Error_test_nn (numpy.ndarray): Tuple of lowest test errors and optimal hidden neurons for neural network classification, for each fold
        - learning_curves (numpy.ndarray): Training errors for neural network classification.
    """

    # Store errors for regression and baseline for each fold
    train_errors_regression = np.empty((K_outer, 2))
    test_errors_regression = np.empty((K_outer, 2))
    train_errors_baseline = np.empty((K_outer, 1))
    test_errors_baseline = np.empty((K_outer, 1))
    train_errors_nn = np.empty((K_outer, 2))
    test_errors_nn = np.empty((K_outer, 2))
    learning_curves = []

    # Store weights for regularized linear regression
    w_rlr_arr = []

    kf_outer = model_selection.KFold(n_splits=K_outer, shuffle=True)

    if regression:
        print("Running regression")

        # outer k-fold loop
        for i, (train_idx, test_idx) in enumerate(kf_outer.split(X, y)):

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            # Training data for inner loop
            train_data = (X_train, y_train)

            # Standardize the features based on the training set. This is only for the neural network
            # The other methods have their own standardization on the inner loop

            scalerX = StandardScaler()
            scalerY = StandardScaler()
            # Standardize the features based on the training set
            X_train_st = scalerX.fit_transform(X_train)
            X_test_st = scalerX.transform(X_test)

            # Standardize the y_train because of the very low variance in the target variable
            y_train_st = scalerY.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_st = scalerY.transform(y_test.reshape(-1, 1)).flatten()

            # outer training data
            outer_train_data = (X_train_st, y_train_st)
            outer_test_data = (X_test_st, y_test_st)

            # Standardize for the neural network
            if nn_standardize:
                outer_train_data_nn = (X_train_st, y_train_st)
                outer_test_data_nn = (X_test_st, y_test_st)

            else:
                outer_train_data_nn = (X_train, y_train)
                outer_test_data_nn = (X_test, y_test)

            # Regalurized linear regression
            (
                opt_lambda,
                opt_val_err,
                mean_w_vs_lambda,
                train_err_vs_lambda,
                test_err_vs_lambda,
            ) = rlr.train_rlr(train_data, alphas=alphas, K_inner=K_inner)

            # Testing regression for the current fold
            test_err, train_err, w_rlr, squared_err_rlr = rlr.outer_test_rlr(
                outer_test_data, outer_train_data, opt_lambda
            )

            train_errors_regression[i] = (train_err, opt_lambda)
            test_errors_regression[i] = (test_err, opt_lambda)
            w_rlr_arr.append(w_rlr)

            # Baseline testing for the current fold
            test_errors_baseline[i], train_errors_baseline[i], squared_err_baseline = (
                rlr.lr_baseline(outer_train_data, outer_test_data)
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

            # Training and testing best model from the inner layer
            test_err, _, train_err, squared_err_nn = nn_reg.outer_test(
                train_set=outer_train_data_nn,
                test_set=outer_test_data_nn,
                hidden_neuron=inner_fold_optimal_hidden_neurons,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )
            # Get last elements form train_err

            test_errors_nn[i] = (test_err, inner_fold_optimal_hidden_neurons)
            train_errors_nn[i] = (train_err[-1], inner_fold_optimal_hidden_neurons)
            learning_curves.append(train_err)

            resuluts_t_test = stats.t_test(
                squared_err_rlr=squared_err_rlr,
                squared_err_nn=squared_err_nn,
                squared_err_baseline=squared_err_baseline,
            )

            # Print progress
            if verbose >= 0:
                print_fold_results(
                    i,
                    K_outer,
                    train_errors_baseline,
                    train_errors_regression,
                    train_errors_nn,
                    test_errors_baseline,
                    test_errors_regression,
                    test_errors_nn,
                )
                print_statistical_test_results(resuluts_t_test, regression)

            # Plot different regeralization rates for the last fold
            if plot and i == K_outer - 1:
                rlr.plot_rlr(
                    alphas,
                    mean_w_vs_lambda,
                    train_err_vs_lambda,
                    test_err_vs_lambda,
                    opt_lambda,
                    index=0,
                )

        return (
            train_errors_regression,
            test_errors_regression,
            train_errors_baseline,
            test_errors_baseline,
            train_errors_nn,
            test_errors_nn,
            learning_curves,
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

            # Standardize the features based on the training set. This is only for the neural network
            # The other methods have their own standardization on the inner loop

            scaler = StandardScaler()

            # Standardize the features based on the training set
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            # outer training data
            outer_train_data = (X_train_st, y_train)
            outer_test_data = (X_test_st, y_test)

            # Standardize for the neural network
            if nn_standardize:
                outer_train_data_nn = (X_train_st, y_train)
                outer_test_data_nn = (X_test_st, y_test)

            else:
                outer_train_data_nn = (X_train, y_train)
                outer_test_data_nn = (X_test, y_test)

            # Baseline testing for the current fold for classification for the dominant class
            test_errors_baseline[i], train_errors_baseline[i], prediction_baseline = (
                log_reg.baseline(outer_train_data, outer_test_data)
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

            test_err, train_err, prediction_nn = nn_class.outer_test(
                outer_train_data_nn,
                outer_test_data_nn,
                hidden_neuron=inner_fold_optimal_hidden_neurons,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            test_errors_nn[i] = (test_err, inner_fold_optimal_hidden_neurons)
            train_errors_nn[i] = (train_err[-1], inner_fold_optimal_hidden_neurons)
            learning_curves.append(train_err)

            # Testing regression for the current fold
            test_err, train_err, opt_alpha_idx, opt_alpha, model, prediction_log_reg = (
                log_reg.train_eval(
                    train_data,
                    outer_train_data,
                    outer_test_data,
                    alphas,
                    K_inner,
                    verbose=verbose,
                )
            )
            test_errors_regression[i] = (test_err, opt_alpha)
            train_errors_regression[i] = (train_err, opt_alpha)

            # Perform statistical tests
            results_mc_menar = stats.mc_nemar(
                y_true=y_test,
                pred_nn=prediction_nn,
                pred_baseline=prediction_baseline,
                pred_log_reg=prediction_log_reg,
                alpha=0.05,
            )

            # Print progress
            if verbose >= 0:
                print_fold_results(
                    i,
                    K_outer,
                    train_errors_baseline,
                    train_errors_regression,
                    train_errors_nn,
                    test_errors_baseline,
                    test_errors_regression,
                    test_errors_nn,
                )
                print_statistical_test_results(results_mc_menar, regression)

        return (
            train_errors_regression,
            test_errors_regression,
            train_errors_baseline,
            test_errors_baseline,
            train_errors_nn,
            test_errors_nn,
            learning_curves,
        )
    # Importing the data


y = np.asanyarray((df["RI"]).squeeze())
X = np.asanyarray(df.drop(["RI"], axis=1))

N, M = X.shape

neurons = [1, 2, 3, 5, 7, 10, 15, 17, 20, 25]
alphas = np.logspace(-5, 9, num=40)

_, _, _, _, _, _, learning_curves = outer_layer(
    X,
    y,
    batch_size=5,
    hidden_neurons_list=neurons,
    epochs=50,
    K_outer=10,
    K_inner=10,
    verbose=0,
    alphas=alphas,
    plot=True,
    nn_standardize=True,
    regression=True,
)

plot_learning_curves(learning_curves)
