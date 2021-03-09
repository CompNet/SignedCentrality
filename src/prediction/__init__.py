#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from path import get_csv_folder_path

"""
This package contains functions related to the classification and regression.
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
    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
    X = df.to_numpy()

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


def process_graphics(Y_test, Y_pred, output):
    """
    Process graphical display for test results
    """

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)

