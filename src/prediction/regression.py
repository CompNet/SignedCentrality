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
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction, \
    perform_prediction


def test_regression(reg, X_test, Y_test, output, print_results=True, export_predicted_values=True, export_graphical_results=True):
    """
    Perform validation tests for regression

    :param reg: trained regression model
    :param X_test: input test data
    :param Y_test: output test data
    """

    prediction_metrics = [
        # metrics.r2_score,  # Best value: 1
        metrics.mean_squared_error,  # Best value: 0
        # metrics.mean_absolute_error  # Best value: 0
    ]

    return test_prediction(reg, X_test, Y_test, output, prediction_metrics, print_results, export_predicted_values, export_graphical_results)


def perform_svr_regression(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
    """This method performs the task of regression for a single output.

    The regression is computed using SVM.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    """

    # Set default values for hyper parameters:
    default_values = {
        "kernel": consts.PREDICTION_KERNEL_LINEAR,
    }

    return perform_prediction(svm.SVR, default_values, features, output, test_regression, print_results, export_predicted_values, export_graphical_results, **kwargs)


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
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    """

    # Set default values for hyper parameters:
    default_values = {
        consts.LinearRegression.FIT_INTERCEPT: True,
        consts.LinearRegression.NORMALIZE: False,
        consts.LinearRegression.COPY_X: True,
        consts.LinearRegression.N_JOBS: -1,
        consts.LinearRegression.POSITIVE: False
    }

    return perform_prediction(LinearRegression, default_values, features, output, test_regression, print_results, export_predicted_values, export_graphical_results, **kwargs)


def perform_mlp_regression(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
    """
    Performs regression using a multilayer perceptron

    :param features: a list of features
    :param output: a single output
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    """

    # Set default values for hyper parameters:
    default_values = {
        consts.MLP.HIDDEN_LAYER_SIZES: (100, ),
        consts.MLP.ACTIVATION: consts.MLP.TANH,
        consts.MLP.SOLVER: consts.MLP.SGD,
        consts.MLP.ALPHA: 0.0001,
        consts.MLP.BATCH_SIZE: consts.MLP.AUTO,
        consts.MLP.LEARNING_RATE: consts.MLP.CONSTANT,
        consts.MLP.LEARNING_RATE_INIT: 0.001,
        consts.MLP.POWER_T: 0.5,
        consts.MLP.MAX_ITER: 15_000,
        consts.MLP.SHUFFLE: True,
        consts.MLP.RANDOM_STATE: None,
        consts.MLP.TOL: 0.0001,
        consts.MLP.VERBOSE: False,
        consts.MLP.WARM_START: False,
        consts.MLP.MOMENTUM: 0.9,
        consts.MLP.NESTEROVS_MOMENTUM: True,
        consts.MLP.EARLY_STOPPING: False,
        consts.MLP.VALIDATION_FRACTION: 0.1,
        consts.MLP.BETA_1: 0.9,
        consts.MLP.BETA_2: 0.999,
        consts.MLP.EPSILON: 1e-08,
        consts.MLP.N_ITER_NO_CHANGE: 10,
        consts.MLP.MAX_FUN: 15_000
    }

    return perform_prediction(MLPRegressor, default_values, features, output, test_regression, print_results, export_predicted_values, export_graphical_results, **kwargs)

