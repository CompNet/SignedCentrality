#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import traceback
from math import nan, isnan
from sys import stderr
from warnings import catch_warnings, filterwarnings
from sklearn.exceptions import ConvergenceWarning
import consts
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import prediction
from collect.collect_predicted_values import collect_predicted_values
import collect.collect_graphics
from path import get_csv_folder_path
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN


"""
This package contains functions related to the classification and regression.

.. note: Authors are given here only for this module.

@author: nejat
@author: Virgile Sucal
"""


def import_data(features, output):
    """
    Import data

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    """

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_full.csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_full.csv"), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1) consts.CSV
    scaler.fit(X)
    X = scaler.transform(X)

    return X, Y


def split_data(X, Y):
    """
    Split data

    :param X: Input data
    :param Y: Output data
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    return X_train, X_test, Y_train, Y_test


def perform_imbalance_correction(X, Y, imbalance_correction_method):
    """
    Perform imbalance correction

    :param X: Input data
    :param Y: Output data
    :return: Corrected X and Y
    """

    # fit and apply the transform
    return imbalance_correction_method.fit_resample(X, Y)


def initialize_data(features, output, imbalance_correction_method=False):
    """
    Initialize input and output sets for training and tests

    :param features: a list of features
    :type features: string list
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :type output: string
    :param imbalance_correction: True if imbalance correction must be performed
    """

    # =======================================================
    # Read features and output from file
    # =======================================================
    X, Y = import_data(features, output)

    # =======================================================
    # Imbalance correction
    # =======================================================
    if imbalance_correction_method:
        X, Y = perform_imbalance_correction(X, Y, imbalance_correction_method=imbalance_correction_method)

    # =======================================================
    # Split data intro train and test sets
    # =======================================================
    return split_data(X, Y)


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


def test_prediction(reg, X_test, Y_test, output, prediction_metrics, print_results=True, export_predicted_values=True, export_graphical_results=True, print_stack_trace=False):
    """
    Perform validation tests

    :param reg: trained prediction model
    :param X_test: input test data
    :param Y_test: output test data
    :param print_stack_trace: True if stack trace must be printed
    """

    # =======================================================
    # Test: Predict the response for test dataset
    # =======================================================
    filterwarnings("error", category=RuntimeWarning)  # To catch runtime warnings as errors.
    Y_pred = None
    try:
        Y_pred = reg.predict(X_test)  # Returns a numpy.ndarray
    except RuntimeWarning:  # If there is an overflow in matmul.
        if print_stack_trace:
            traceback.print_exc()
            print(file=stderr)
            print("RuntimeWarning has been caught in 'test_prediction()'.", file=stderr)
            print(file=stderr)
    filterwarnings("default")  # To reset warnings default behavior.

    # =======================================================
    # Metrics
    # =======================================================
    prediction_metrics_results = {}
    for metric in prediction_metrics:
        # if sum([int(isnan(y)) for y in Y_pred]) > 0:  # y may be NAN if there has been an overflow error in model.fit()
        #     prediction_metrics_results[metric.__name__] = float('nan')
        #     continue
        try:
            prediction_metrics_results[metric.__name__] = metric(Y_test, Y_pred)
        except:
            if print_stack_trace:
                traceback.print_exc()
                if Y_pred is not None and sum([int(isnan(y)) for y in Y_pred]) > 0:  # y may be NAN if there has been an overflow in model.fit()
                    print("Error: NAN in Y_pred.", file=stderr)
                elif Y_pred is None:
                    print("Error: Y_pred is None.", file=stderr)
                else:
                    print("Error: Unknown error has occurred.", file=stderr)
                print("\tMetric value is NAN.", file=stderr)
            prediction_metrics_results[metric.__name__] = float('nan')

    if print_results:
        for metric in prediction_metrics:
            print(metric.__name__ + ":", prediction_metrics_results[metric.__name__])

    # Save predicted values into a file
    if export_predicted_values:
        collect_predicted_values(Y_pred, output)

    # Save graphics into a file
    if export_graphical_results:
        process_graphics(Y_test, Y_pred, output)

    return prediction_metrics_results


def perform_prediction(model_class, default_values, features, output, test_function, print_results=True, export_predicted_values=True, export_graphical_results=True, print_stack_trace=False, imbalance_correction_method=False, **kwargs):
    """This method performs the task of prediction for a single output.

    :param model_class: prediction technique
    :param default_values: default values for hyper parameters
    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param test_function: function to perform validation tests and compute metrics
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    :param print_stack_trace: True if stack trace must be printed
    """

    # =======================================================
    #  Initialization
    # =======================================================
    X_train, X_test, Y_train, Y_test = initialize_data(features, output, imbalance_correction_method=imbalance_correction_method)
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    # =======================================================
    #  Train: Create a the predictor
    # =======================================================
    model = model_class(**hyper_parameters)
    filterwarnings("error", category=RuntimeWarning)  # To catch runtime warnings as errors.
    filterwarnings("ignore", category=ConvergenceWarning)
    try:
        model.fit(X_train, Y_train)  # There is an overflow in matmul if there are 300 layers, activation function is "identity" and solver is 'sgd'.
    except RuntimeWarning:  # If there is an overflow in matmul.
        if print_stack_trace:
            traceback.print_exc()
            print(file=stderr)
            print("RuntimeWarning has been caught in 'perform_prediction()'.", file=stderr)
            for k, v in kwargs.items():
                if k == consts.MLP.HIDDEN_LAYER_SIZES:
                    print("len(", k, ")", ":", len(v), file=stderr)
                print(k, ": ", v, sep="", file=stderr)
            print(file=stderr)
    filterwarnings("default")  # To reset warnings default behavior.

    # =======================================================
    #  Tests
    # =======================================================
    computed_prediction_metrics = test_function(model, X_test, Y_test, output[0], print_results, export_predicted_values, export_graphical_results)  # A matmul overflow error cause exit here.

    return model, computed_prediction_metrics


def process_graphics(Y_test, Y_pred, output):
    """
    Process graphical display for test results
    """

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)
    collect.collect_graphics.generate_boxplot_clean(Y_test, Y_pred, output)

