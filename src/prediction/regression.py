#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related regression computing.

.. note: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
.. note: https://scikit-learn.org/stable/modules/metrics.html#linear-kernel
.. note: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

@author: nejat
@author: Virgile Sucal
"""

from sys import stderr
from deprecated import deprecated
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import consts
from prediction import initialize_hyper_parameters, initialize_data, process_graphics


def test_regression(reg, X_test, Y_test):
    """
    Perform validation tests

    :param reg: trained regression model
    :param X_test: input test data
    :param Y_test: output test data
    """

    # =======================================================
    # Test: Predict the response for test dataset
    # =======================================================
    Y_pred = reg.predict(X_test)  # Returns a numpy.ndarray

    # =======================================================
    # Metrics
    # =======================================================
    # print("Test dataset:", Y_test)
    # print("Predicted dataset:", Y_pred)
    print("R2 score:", metrics.r2_score(Y_test, Y_pred))  # Best value: 1
    print("Mean squared error:", metrics.mean_squared_error(Y_test, Y_pred))  # Best value: 0
    print("Mean absolute error:", metrics.mean_absolute_error(Y_test, Y_pred), "\n")  # Best value: 0

    process_graphics(Y_test, Y_pred)


def perform_svr_regression(features, output, **kwargs):
    """This method performs the task of regression for a single output.

    The regression is computed using SVM.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    """

    # =======================================================
    #  Initialization
    # =======================================================
    X_train, X_test, Y_train, Y_test = initialize_data(features, output)

    # Set default values for hyper parameters:
    default_values = {
        "kernel": consts.PREDICTION_KERNEL_LINEAR,
    }
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    # =======================================================
    #  Train: Create a svm Regressor
    # =======================================================
    # >> other params: gamma, max_iter, degree, shrinking
    reg = svm.SVC(**hyper_parameters)
    reg.fit(X_train, Y_train)

    # =======================================================
    #  Tests
    # =======================================================
    test_regression(reg, X_test, Y_test)


@deprecated("This function is deprecated, use 'perform_svr_regression()' instead")
def perform_regression(features, output, kernel):
    """
    Alias for perform_svr_regression().

    This function is deprecated, use 'perform_svr_regression()' instead
    It hasn't been deleted for compatibility with previous versions.
    It should not be used in new functions.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    """

    return perform_svr_regression(features, output, kernel=kernel)


def perform_linear_regression(features, output, **kwargs):
    """
    Performs linear regression

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    """

    # =======================================================
    #  Initialization
    # =======================================================
    X_train, X_test, Y_train, Y_test = initialize_data(features, output)

    # Set default values for hyper parameters:
    default_values = {
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "n_jobs": -1,
        "positive": False
    }
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    # =======================================================
    #  Train: Create a Linear Regressor
    # =======================================================
    model = LinearRegression(**hyper_parameters).fit(X_train, Y_train)

    # =======================================================
    #  Tests
    # =======================================================
    test_regression(model, X_test, Y_test)


def perform_mlp_regression(features, output, **kwargs):
    """
    Performs regression using a multilayer perceptron

    :param features: a list of features
    :param output: a single output
    """

    # =======================================================
    #  Initialization
    # =======================================================
    X_train, X_test, Y_train, Y_test = initialize_data(features, output)

    # Set default values for hyper parameters:
    default_values = {
        "hidden_layer_sizes": 20000,
        "activation": consts.MLP.TANH,
        "solver": consts.MLP.SGD,
        "alpha": 0.0001,
        "batch_size": consts.MLP.AUTO,
        "learning_rate": consts.MLP.CONSTANT,
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 15_000,
        "shuffle": False,
        "random_state": None,
        "tol": 0.0001,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "n_iter_no_change": 10,
        "max_fun": 15_000
    }
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    # =======================================================
    #  Train: Create a MLP Regressor
    # =======================================================
    print("Train...", file=stderr)
    model = MLPRegressor(**hyper_parameters).fit(X_train, Y_train)
    print("Training done.", file=stderr)

    # =======================================================
    #  Tests
    # =======================================================
    test_regression(model, X_test, Y_test)
