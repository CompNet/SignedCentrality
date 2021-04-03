#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related classification computing.

.. note: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
.. note: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
.. note: https://scikit-learn.org/stable/modules/metrics.html#linear-kernel
.. note: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
.. note: https://scikit-learn.org/stable/modules/model_evaluation.html#clustering-metrics

@author: nejat
@author: Virgile Sucal
@author: Laurent Pereira
"""

from sys import stderr
from deprecated import deprecated
from sklearn import metrics
from sklearn import svm
import consts
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction, \
    perform_prediction



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


def perform_svc_classification(features, output, print_results=True, export_predicted_values=True, export_graphical_results=False, imbalance_correction_method=False, **kwargs):
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
        "kernel": consts.PREDICTION_KERNEL_LINEAR,
    }

    return perform_prediction(svm.SVC, default_values, features, output, test_classification, print_results, export_predicted_values, export_graphical_results, imbalance_correction_method=imbalance_correction_method, **kwargs)


@deprecated("This function is deprecated, use 'perform_svc_classification()' instead")
def perform_classification(features, output, kernel, imbalance_correction_method=False):
    """
    Alias for perform_svc_classification().

    This function is deprecated, use 'perform_svc_classification()' instead
    It hasn't been deleted for compatibility with previous versions.
    It should not be used in new functions.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    """

    return perform_svc_classification(features, output, kernel=kernel, imbalance_correction_method=imbalance_correction_method)

