'''
Created on Mar 04, 2021

@author: nejat
@author: Virgile Sucal
@author: Laurent Pereira
@author: Alexandre
'''

import os
import consts
import path

import pandas as pd
import numpy as np
from sklearn import *
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sys import stderr
from deprecated import deprecated
from sklearn import metrics
from sklearn import svm
import consts
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction, \
    perform_prediction

import collect.collect_graphics


def test_classification(cla, X_test, Y_test, output, print_results=True, export_predicted_values=True, export_graphical_results=True):
    """
    Perform validation tests for classification

    :param cla: trained classification model
    :param X_test: input test data
    :param Y_test: output test data
    """

    prediction_metrics = [
        metrics.f1_score,  # Best value: 1
        metrics.accuracy_score,  # Best value: 1 if normalize == True, else the number of correctly classified samples
        metrics.precision_score,  # Best value: 1
        metrics.recall_score  # Best value: 1
    ]

    return test_prediction(cla, X_test, Y_test, output, prediction_metrics, print_results, export_predicted_values, export_graphical_results)


def perform_random_forest_classification(features, output, print_results=True, export_predicted_values=True, export_graphical_results=True, imbalance_correction_method=False, **kwargs):
    """This method performs the task of classification for a single output.

    The classification is computed using SVM.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
    :param print_results: True if metrics results must be printed
    :param export_predicted_values: True if predicted values must be exported
    :param export_graphical_results: True if graphical results must be exported
    """

    # Set default values for hyper parameters:
    default_values = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf" : 1,
    }

    return perform_prediction(RandomForestClassifier, default_values, features, output, test_classification, print_results, export_predicted_values,
                              export_graphical_results, imbalance_correction_method=imbalance_correction_method, **kwargs)


@deprecated("This function is deprecated, use 'perform_svc_classification()' instead")
def perform_classification(features, output, imbalance_correction_method=False, n_estimators=1000, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Alias for perform_svc_classification().

    This function is deprecated, use 'perform_svc_classification()' instead
    It hasn't been deleted for compatibility with previous versions.
    It should not be used in new functions.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    """

    return perform_random_forest_classification(features, output, imbalance_correction_method=imbalance_correction_method, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

