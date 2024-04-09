import numpy as np
import scipy.stats
import scipy.stats as st
from dtuimldmtools import mcnemar


def t_test(squared_err_rlr, squared_err_nn, squared_err_baseline, alpha=0.05):
    """
    Perform a t-test to compare the models.

    Args:
        squared_err_rlr (array-like): Squared errors for model A.
        squared_err_nn (array-like): Squared errors for model B.
        squared_err_baseline (array-like): Squared errors for model C.
        alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the confidence intervals for model A (CIA_A),
               model B (CIA_B), and the difference between model C and model A (CI).
    """

    # compute confidence interval of model A
    CI_rlr = st.t.interval(
        1 - alpha,
        df=len(squared_err_rlr) - 1,
        loc=np.mean(squared_err_rlr),
        scale=st.sem(squared_err_rlr),
    )  # Confidence interval

    # compute confidence interval of model B
    CI_nn = st.t.interval(
        1 - alpha,
        df=len(squared_err_nn) - 1,
        loc=np.mean(squared_err_nn),
        scale=st.sem(squared_err_nn),
    )

    CIA_baseline = st.t.interval(
        1 - alpha,
        df=len(squared_err_baseline) - 1,
        loc=np.mean(squared_err_baseline),
        scale=st.sem(squared_err_baseline),
    )

    # compute confidence interval bw model A and model B
    z_rlr_baseline = squared_err_rlr - squared_err_baseline
    CI_rlr_baseline = st.t.interval(
        1 - alpha,
        len(z_rlr_baseline) - 1,
        loc=np.mean(z_rlr_baseline),
        scale=st.sem(z_rlr_baseline),
    )  # Confidence interval
    p_rlr_baseline = 2 * st.t.cdf(
        -np.abs(np.mean(z_rlr_baseline)) / st.sem(z_rlr_baseline),
        df=len(z_rlr_baseline) - 1,
    )  # p-value

    # Compute the confidence interval between model A and B
    z_rlr_nn = squared_err_rlr - squared_err_nn
    CI_rlr_nn = st.t.interval(
        1 - alpha, len(z_rlr_nn) - 1, loc=np.mean(z_rlr_nn), scale=st.sem(z_rlr_nn)
    )  # Confidence interval
    p_rlr_nn = 2 * st.t.cdf(
        -np.abs(np.mean(z_rlr_nn)) / st.sem(z_rlr_nn), df=len(z_rlr_nn) - 1
    )  # p-value

    # Compute the copnfidence interval between model B and C
    z_nn_baseline = squared_err_nn - squared_err_baseline
    CI_nn_baseline = st.t.interval(
        1 - alpha,
        len(z_nn_baseline) - 1,
        loc=np.mean(z_nn_baseline),
        scale=st.sem(z_nn_baseline),
    )  # Confidence interval
    p_nn_baseline = 2 * st.t.cdf(
        -np.abs(np.mean(z_nn_baseline)) / st.sem(z_nn_baseline),
        df=len(z_nn_baseline) - 1,
    )  # p-value

    return {
        "rlr": CI_rlr,
        "nn": CI_nn,
        "baseline": CIA_baseline,
        "rlr_baseline": (CI_rlr_baseline, p_rlr_baseline),
        "rlr_nn": (CI_rlr_nn, p_rlr_nn),
        "nn_baseline": (CI_nn_baseline, p_nn_baseline),
    }


def mc_nemar(y_true, pred_nn, pred_baseline, pred_log_reg, alpha=0.05):
    """
    Perform a McNemar test to compare two models.

    Args:
        y_true (array-like): True labels.
        pred_nn (array-like): Predicted labels from model 1.
        pred_baseline (array-like): Predicted labels from model 2.
        alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the point estimate (thetahat), confidence interval (CI), and p-value (p).
    """
    print(f"baseline: {pred_baseline}")
    print(f"nn: {pred_nn}")
    print(f"log_reg: {pred_log_reg}")
    print(f"y_true: {y_true}")

    [thetahat_1, CI_1, p_1] = mcnemar(
        y_true, pred_nn, pred_baseline, alpha=alpha, print=False
    )

    [thetahat_2, CI_2, p_2] = mcnemar(
        y_true, pred_nn, pred_log_reg, alpha=alpha, print=False
    )

    [thetahat_3, CI_3, p_3] = mcnemar(
        y_true, pred_log_reg, pred_baseline, alpha=alpha, print=False
    )

    return {
        "nn_baseline": [thetahat_1, CI_1, p_1],
        "nn_logreg": [thetahat_2, CI_2, p_2],
        "baseline_logreg": [thetahat_3, CI_3, p_3],
    }
