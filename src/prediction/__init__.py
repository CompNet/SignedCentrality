#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def initialize_data(features, output):
    """
    Initialize input and output sets for training and tests

    :param features: a list of features
    :type features: string list
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :type output: string
    """

    # =======================================================
    # Read features and output from file
    # =======================================================

##    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
##    Y = df.to_numpy()
##
##    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
##    X = df.to_numpy()
    
    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_full.csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_full.csv"), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)

    
##    #Rectify the imbalance in the data UNDERSAMPLING
##    undersample = RandomUnderSampler(sampling_strategy='majority')
##    undersample = NearMiss(version=1)
##    undersample = NearMiss(version=2, n_neighbors=3)
##    undersample = NearMiss(version=3, n_neighbors_ver3=3)
##    undersample = CondensedNearestNeighbour(n_neighbors=1)
##    undersample = TomekLinks()
##    undersample = EditedNearestNeighbours(n_neighbors=3)
##    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
##    undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)

##    # fit and apply the transform
##    X, Y = undersample.fit_resample(X, Y)


##    #Rectify the imbalance in the data OVERSAMPLING
##    oversample = RandomOverSampler(sampling_strategy='minority')
##    oversample = SMOTE()
##    oversample = BorderlineSMOTE()
##    oversample = SVMSMOTE()
    oversample = ADASYN()

    X, Y = oversample.fit_resample(X, Y)

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


def test_prediction(reg, X_test, Y_test, output, prediction_metrics, print_results=True, export_predicted_values=True, export_graphical_results=True):
    """
    Perform validation tests

    :param reg: trained prediction model
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
    prediction_metrics_results = {}
    for metric in prediction_metrics:
        prediction_metrics_results[metric.__name__] = metric(Y_test, Y_pred)

    if print_results:
        for metric in prediction_metrics:
            print(metric.__name__ + ":", prediction_metrics_results[metric.__name__])

    # Save predicted values into a file
##    if export_predicted_values:
##        collect_predicted_values(Y_pred, output)

    # Save graphics into a file
    if export_graphical_results:
        process_graphics(Y_test, Y_pred, output)

    return prediction_metrics_results


def perform_prediction(model_class, default_values, features, output, test_function, print_results=True, export_predicted_values=True, export_graphical_results=True, **kwargs):
    """This method performs the task of prediction for a single output.

    :param model_class: prediction technique
    :param default_values: default values for hyper parameters
    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param test_function: function to perform validation tests and compute metrics
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    """

    # =======================================================
    #  Initialization
    # =======================================================
    X_train, X_test, Y_train, Y_test = initialize_data(features, output)
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    # =======================================================
    #  Train: Create a the predictor
    # =======================================================
    model = model_class(**hyper_parameters)
    model.fit(X_train, Y_train)

    # =======================================================
    #  Tests
    # =======================================================
    computed_prediction_metrics = test_function(model, X_test, Y_test, output[0], print_results, export_predicted_values, export_graphical_results)

    return model, computed_prediction_metrics


def process_graphics(Y_test, Y_pred, output):
    """
    Process graphical display for test results
    """

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)
    collect.collect_graphics.generate_boxplot_clean(Y_test, Y_pred, output)

