#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import path
from path import get_csv_folder_path

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

import collect.collect_graphics
import collect.collect_predicted_values
import prediction.classification
import prediction.regression
from prediction import initialize_hyper_parameters, initialize_data, process_graphics, test_prediction, \
    perform_prediction


"""
Temporary version of feature ablation, to test the "same metric score for classification" issue

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


def perform_feature_ablation_eq_sol(predictor, default_values, features, output, prediction_name, is_classifier=True, **kwargs):
    """This method is the general method to perform the task of feature ablation for a single output.

        :param predictor: prediction technique
        :param default_values: default values for hyper parameters
        :param features: a list of features
        :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
        :param prediction_name : name of the predictor, used only for saving plot
        :param is_classifier : a simple boolean saying if it's a classifier or a regresser
        """

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_eq_sol" + consts.CSV), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_eq_sol" + consts.CSV), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1) consts.CSV
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    model = predictor(**hyper_parameters)
    rfe = RFE(estimator=model)
    rfe.fit(X_train, Y_train)
    ranking = rfe.ranking_

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

    collect.collect_graphics.generate_plot(feature_list, scores, "feature_ablation_"+str(prediction_name))


def perform_feature_ablation_eq_solclass(predictor, default_values, features, output, prediction_name, is_classifier=True, **kwargs):
    """This method is the general method to perform the task of feature ablation for a single output.

        :param predictor: prediction technique
        :param default_values: default values for hyper parameters
        :param features: a list of features
        :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
        :param prediction_name : name of the predictor, used only for saving plot
        :param is_classifier : a simple boolean saying if it's a classifier or a regresser
        """

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_eq_solclass" + consts.CSV), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_eq_solclass" + consts.CSV), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1) consts.CSV
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    hyper_parameters = initialize_hyper_parameters(default_values, kwargs)

    model = predictor(**hyper_parameters)
    rfe = RFE(estimator=model)
    rfe.fit(X_train, Y_train)
    ranking = rfe.ranking_

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

    collect.collect_graphics.generate_plot(feature_list, scores, "feature_ablation_"+str(prediction_name))


def feature_ablation_svc_classification_eq_sol(features, output, is_classifier=True, **kwargs):
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

    return perform_feature_ablation_eq_sol(svm.SVC, default_values, features, output, "SVC_eq_sol", is_classifier, **kwargs)

def feature_ablation_svc_classification_eq_solclass(features, output, is_classifier=True, **kwargs):
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

    return perform_feature_ablation_eq_solclass(svm.SVC, default_values, features, output, "SVC_eq_solclass", is_classifier, **kwargs)