'''
Created on Sep 23, 2020

@author: nejat
'''
import os
import itertools
from sys import stderr

import consts
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from path import get_csv_folder_path
from deprecated import deprecated

import collect.collect_graphics

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# https://scikit-learn.org/stable/modules/metrics.html#linear-kernel
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics


def initialize_data(features, output):
    """
    Initialize input and output sets for training and tests

    :param features: a list of features
    :type features: string list
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :type output: string
    """

    # =======================================================
    # Read features and output from file (original code)
    # =======================================================
    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
    X = df.to_numpy()

    # =======================================================
    # Read features and output from file (test code)
    # =======================================================
    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_full.csv"), usecols=output)
    # Y = df.to_numpy()

    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_full.csv"), usecols=features)
    # X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)

    # =======================================================
    # Split data intro train and test sets
    # =======================================================
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    return X_train, X_test, Y_train, Y_test


def initialize_hyper_parameters(default_values, user_defined_values):
    """
    Compute hyper parameters merging default values and user defined values

    :param default_values: default values for hyper parameters
    :param user_defined_values: user defined values for hyper parameters
    :return: merged hyper parameters
    """

    hyper_parameters = {}
    for key, value in default_values.items():
        param_value = value
        if key in user_defined_values:
            param_value = user_defined_values[key]
        hyper_parameters = {**hyper_parameters, key: param_value}

    return hyper_parameters



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
    Y_pred = reg.predict(X_test) # Returns a numpy.ndarray


    # print("Predicted dataset before rounding:", Y_pred) # I want to transform decimal values to integer values
    # i = 0
    # for val in Y_pred:
    #     print("Value before rounding:", val)
    #     Y_pred[i] = round(val)
    #     print("Value after rounding:",Y_pred[i])
    #     i += 1
    # print("Predicted dataset after rounding:", Y_pred)

    # =======================================================
    # Metrics
    # =======================================================
    # print("Test dataset:", Y_test)
    # print("Predicted dataset:", Y_pred)
    print("R2 score:", metrics.r2_score(Y_test, Y_pred))  # Best value: 1
    print("Mean squared error:", metrics.mean_squared_error(Y_test, Y_pred))  # Best value: 0
    print("Mean absolute error:", metrics.mean_absolute_error(Y_test, Y_pred), "\n")  # Best value: 0

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)


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


