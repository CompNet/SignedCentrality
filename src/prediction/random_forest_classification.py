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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection

import collect.collect_graphics


def test_classification(cla, X_test, Y_test, output, print_results=True, export_predicted_values=True, export_graphical_results=False):
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


def perform_random_forest_classification(features, output, print_results=True, export_predicted_values=True, export_graphical_results=False, **kwargs):
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
        "n_estimators": 10000,
    }

    return perform_prediction(RandomForestClassifier, default_values, features, output, test_classification, print_results, export_predicted_values, export_graphical_results, **kwargs)


@deprecated("This function is deprecated, use 'perform_svc_classification()' instead")
def perform_classification(features, output, n_estimators):
    """
    Alias for perform_svc_classification().

    This function is deprecated, use 'perform_svc_classification()' instead
    It hasn't been deleted for compatibility with previous versions.
    It should not be used in new functions.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    """

    return perform_random_forest_classification(features, output, n_estimators=n_estimators)




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
##    oversample = RandomOverSampler(sampling_strategy='majority')
##    oversample = SMOTE()
##    oversample = BorderlineSMOTE()
##    oversample = SVMSMOTE()
##    oversample = ADASYN()

##    X, Y = oversample.fit_resample(X, Y)
