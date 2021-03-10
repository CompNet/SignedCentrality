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
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction


def test_regression(reg, X_test, Y_test, output, print_results=True, export_predicted_values=True, export_graphical_results=True):
    """
    Perform validation tests for regression

    :param reg: trained regression model
    :param X_test: input test data
    :param Y_test: output test data
    """

    prediction_metrics = [
        metrics.r2_score,  # Best value: 1
        metrics.mean_squared_error,  # Best value: 0
        metrics.mean_absolute_error  # Best value: 0
    ]

    return test_prediction(reg, X_test, Y_test, output, prediction_metrics, print_results, export_predicted_values, export_graphical_results)


def perform_svr_regression(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
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
    computed_regression_metrics = test_regression(reg, X_test, Y_test, output, print_results, export_predicted_values, export_graphical_results)

    return reg, computed_regression_metrics


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


def perform_linear_regression(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
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
    computed_regression_metrics = test_regression(model, X_test, Y_test, output, print_results, export_predicted_values, export_graphical_results)

    return model, computed_regression_metrics


def perform_mlp_regression(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
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
    computed_regression_metrics = test_regression(model, X_test, Y_test, output, print_results, export_predicted_values, export_graphical_results)

    return model, computed_regression_metrics
