#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import svm model
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import EditedNearestNeighbours

# import collect.collect_graphics
import collect.collect_predicted_values
import prediction.classification
import prediction.regression
from collect import collect_graphics
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction, \
    perform_prediction


"""
This module contains functions related feature ablation.

https://www.samueltaylor.org/articles/feature-importance-for-any-model.html
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination

@author: Laurent Pereira
"""


def score_model_classification(predictor, X_train, X_test, Y_train, Y_test):
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)
    # print(metrics.f1_score(Y_test, Y_pred))
    return metrics.f1_score(Y_test, Y_pred)


def score_model_regression(predictor, X_train, X_test, Y_train, Y_test):
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)
    # print(metrics.mean_squared_error(Y_test, Y_pred))
    return metrics.mean_squared_error(Y_test, Y_pred)


def perform_feature_ablation(predictor, default_values, features, output, prediction_name, is_classifier=True, imbalance_correction_method=False, **kwargs):
    """This method is the general method to perform the task of feature ablation for a single output.

        :param predictor: prediction technique
        :param default_values: default values for hyper parameters
        :param features: a list of features
        :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
        :param prediction_name : name of the predictor, used only for saving plot
        :param is_classifier : a simple boolean saying if it's a classifier or a regresser
        """

    if is_classifier:
        X_train, X_test, Y_train, Y_test = initialize_data(features, output, imbalance_correction_method)
    if not is_classifier:
        X_train, X_test, Y_train, Y_test = initialize_data(features, output)
    # X_train, X_test, Y_train, Y_test = initialize_data(features, output)
    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    model = predictor(**hyper_parameters)
    rfe = RFE(estimator=model)
    rfe.fit(X_train, Y_train)
    ranking = rfe.ranking_
    print(features)
    print(ranking)  # Contains the rankings for each feature

    base_score = 0
    if is_classifier:
        base_score = score_model_classification(model, X_train, X_test, Y_train, Y_test)
    if not is_classifier:
        base_score = score_model_regression(model, X_train, X_test, Y_train, Y_test)

    features_updated = []  # copy of "features" variable, need to be done to avoid some issues later
    for i in range(0, len(features), 1):
        features_updated.append(features[i])
    ranking_updated = ranking.tolist()  # Convert numpy.ndarray to list (list.pop() will be used later)

    scores = []  # list that will contains all scores
    feature_list = []  # list that will contains all features that were deleted
    use_column = []  # list that will contains boolean values, to select which column will be used
    for i in range(len(ranking_updated)):
        use_column.append(True)

    scores.append(base_score)
    feature_list.append("Base score")
    # print(scores)

    while len(ranking_updated) != 1:
        worst_ranking = max(ranking_updated) # searching the feature with the worst rank
        index_worst_ranking = ranking_updated.index(max(ranking_updated)) # searching the index of the corresponding ranking
        feature_name_worst_ranking = features_updated[index_worst_ranking] # searching the feature name

        feature_list.append("W/O "+feature_name_worst_ranking)
        # print(feature_list)

        use_column[features.index(feature_name_worst_ranking)] = False  # the index where the feature is found is set to False : this feature will not be used anymore
        # print(features)
        # print(use_column)

        # print(X_train[:, use_column])
        # print(...)
        # print(X_test[:, use_column])

        if is_classifier:
            scores.append(score_model_classification(model,
                                      X_train[:, use_column],
                                      X_test[:, use_column],
                                      Y_train,
                                      Y_test))  # Calculating score with only the features not deleted yet
        if not is_classifier:
            scores.append(score_model_regression(model,
                                                     X_train[:, use_column],
                                                     X_test[:, use_column],
                                                     Y_train,
                                                     Y_test))  # Calculating score with only the features not deleted yet

        # print(scores)

        ranking_updated.pop(index_worst_ranking)  # Deleting ranking associated to the worst feature
        features_updated.pop(index_worst_ranking)  # Deleting worst feature from feature list
        # print(ranking_updated)
        # print(features_updated)

    """if len(ranking_updated) == 1:
        use_column[features.index(features_updated[0])] = False
        # print(use_column)
        feature_list.append("Without " + features_updated[0])
        # print(feature_list)
        scores.append(0)
        # print(scores)"""

    print(feature_list)
    print(scores)
    print("feature ramaining : "+features_updated[0])

    # setting Y axis max and min values
    for it_metric_value in scores:
        if it_metric_value < collect_graphics.Y_MIN:
            collect_graphics.Y_MIN = it_metric_value
        if it_metric_value > collect_graphics.Y_MAX:
            collect_graphics.Y_MAX = it_metric_value

    collect.collect_graphics.generate_plot(feature_list, scores, "feature_ablation_"+str(prediction_name))

    # resetting Y axis max and min values to their original values
    collect_graphics.reset_y_lims()


def feature_ablation_svc_classification(features, output, is_classifier=True, imbalance_correction_method=EditedNearestNeighbours(n_neighbors=3), **kwargs):
    """This method performs the task of feature ablation for a single output using a classifier.

        The classifier used is svm.SVC.

        :param features: a list of features
        :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
        :param is_classifier : a simple boolean saying if it's a classifier or a regresser
        """

    # Set default values for hyper parameters:
    default_values = {
        "kernel": consts.PREDICTION_KERNEL_LINEAR,
    }

    return perform_feature_ablation(svm.SVC, default_values, features, output, "SVC", is_classifier, imbalance_correction_method, **kwargs)

def feature_ablation_random_forest_classification(features, output, is_classifier=True, imbalance_correction_method=EditedNearestNeighbours(n_neighbors=3), **kwargs):
    """This method performs the task of feature ablation for a single output using a classifier.

    The classifier used is RandomForestClassifier.

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_IS_SINGLE_SOLUTION
    :param is_classifier : a simple boolean saying if it's a classifier or a regresser
    """
    # Set default values for hyper parameters:
    default_values = {
        "n_estimators": 10000,
    }

    return perform_feature_ablation(RandomForestClassifier, default_values, features, output, "Random_Forest", is_classifier, imbalance_correction_method, **kwargs)


def feature_ablation_svr_regression(features, output, is_classifier=False, **kwargs):
    """Performs feature ablation using svm.SVR for regression

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param is_classifier : a simple boolean saying if it's a classifier or a regresser
    """

    # Set default values for hyper parameters:
    default_values = {
        "kernel": consts.PREDICTION_KERNEL_LINEAR,
    }

    return perform_feature_ablation(svm.SVC, default_values, features, output, "SVR", is_classifier, **kwargs)


def feature_ablation_linear_regression(features, output, is_classifier=False, **kwargs):
    """
    Performs feature ablation for linear regression

    :param features: a list of features
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :param is_classifier : a simple boolean saying if it's a classifier or a regresser
    """

    # Set default values for hyper parameters:
    default_values = {
        consts.LinearRegression.FIT_INTERCEPT: True,
        consts.LinearRegression.NORMALIZE: False,
        consts.LinearRegression.COPY_X: True,
        consts.LinearRegression.N_JOBS: -1,
        consts.LinearRegression.POSITIVE: False
    }

    return perform_feature_ablation(LinearRegression, default_values, features, output, "LinearRegression", is_classifier, **kwargs)


def feature_ablation_mlp_regression(features, output, is_classifier=False, **kwargs): # TODO doesn't work, I need an attribute to select which features are more important than the others
    """
    Performs feature ablation using a multilayer perceptron

    :param features: a list of features
    :param output: a single output
    :param is_classifier : a simple boolean saying if it's a classifier or a regresser
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

    return perform_feature_ablation(MLPRegressor, default_values, features, output, "MLPRegression", is_classifier, **kwargs)

