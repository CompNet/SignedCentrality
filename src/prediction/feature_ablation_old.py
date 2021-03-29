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

import collect.collect_graphics
import collect.collect_predicted_values


"""
Local test for feature ablation for SVC classification only, will use new prediction structure once it works
https://www.samueltaylor.org/articles/feature-importance-for-any-model.html
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination
@author: Laurent Pereira
"""


def feature_ablation(features, output, kernel):
    # =======================================================
    # Read features and output from file (original code)
    # =======================================================
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    base_score = score_model(X_train, X_test, Y_train, Y_test, kernel)
    scores = []
    for i in range(X_train.shape[1]):
        use_column = [ndx != i for ndx in range(X_train.shape[1])]
        scores.append(score_model(X_train[:, use_column],
                                  X_test[:, use_column],
                                  Y_train,
                                  Y_test,
                                  kernel))
        print(scores)

    print(sorted(enumerate([base_score - s for s in scores]),
           key=lambda ndx_score: ndx_score[1],
           reverse=True)[:25])

    # collect.collect_graphics.generate_plot(features, scores, "feature_ablation_test")


def score_model(X_train, X_test, y_train, y_test, kernel):  # the "identical score everytime" issue is probably here
    # predictor = svm.SVC(kernel=kernel)
    predictor = svm.SVR(kernel=kernel)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    # print(metrics.f1_score(y_test, y_pred))
    # return metrics.f1_score(y_test, y_pred)
    print(metrics.mean_squared_error(y_test, y_pred))
    return metrics.mean_squared_error(y_test, y_pred)


def feature_ablation_1(features, output, kernel): # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
    # =======================================================
    # Read features and output from file
    # =======================================================
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    # =======================================================
    # Processing features ranking (SVC only)
    # =======================================================
    # predictor = svm.SVC(kernel=kernel)
    predictor = svm.SVR(kernel=kernel)
    rfe = RFE(estimator=predictor)
    rfe.fit(X_train, Y_train)
    ranking = rfe.ranking_
    # print(ranking) # Contains the rankings for each feature
    # print(features)

    # base_score = score_model(X_train, X_test, Y_train, Y_test, kernel)
    base_score = score_model1(predictor, X_train, X_test, Y_train, Y_test, kernel)

    # =======================================================
    # Initializing variables
    # =======================================================
    features_updated = [] # copy of "features" variable, need to be done to avoid some issues later
    for i in range(0, len(features), 1):
        features_updated.append(features[i])
    ranking_updated = ranking.tolist()  # Convert numpy.ndarray to list (list.pop() will be used later)

    """print(ranking_updated)
    print(type(ranking_updated))
    print(features_updated)
    print(type(features_updated))
    print(features_updated[0])"""

    scores = []  # list that will contains all scores
    feature_list = []  # list that will contains all features that were deleted
    use_column = []  # list that will contains boolean values, to select which column will be used
    for i in range(len(ranking_updated)):
        use_column.append(True)
    # print(use_column)
    scores.append(base_score)
    feature_list.append("Base score")

    print(scores)
    # print(feature_list)
    # print(X_train)
    # print(len(ranking_updated))

    while len(ranking_updated) != 1:
        worst_ranking = max(ranking_updated) # searching the feature with the worst rank
        index_worst_ranking = ranking_updated.index(max(ranking_updated)) # searching the index of the corresponding ranking
        feature_name_worst_ranking = features_updated[index_worst_ranking] # searching the feature name
        # print(worst_ranking)
        # print(index_worst_ranking)
        # print(feature_name_worst_ranking)

        feature_list.append("Without "+feature_name_worst_ranking)
        print(feature_list)

        use_column[features.index(feature_name_worst_ranking)] = False  # the index where the feature is found is set to False : this feature will not be used anymore
        # print(features)
        # print(use_column)

        X_train_reduced = X_train[:, use_column]
        print(X_train_reduced)
        # print(X_train[:, use_column])
        print(...)
        X_test_reduced = X_test[:, use_column]
        print(X_test_reduced)
        # print(X_test[:, use_column])

        scores.append(score_model(X_train[:, use_column],
                                  X_test[:, use_column],
                                  Y_train,
                                  Y_test,
                                  kernel))  # Calculating score with only the features not deleted yet

        """scores.append(score_model1(predictor,
                                  X_train_reduced,
                                  X_test_reduced,
                                  Y_train,
                                  Y_test,
                                  kernel))  # Calculating score with only the features not deleted yet"""

        print(scores)

        ranking_updated.pop(index_worst_ranking)  # Deleting ranking associated to the worst feature
        features_updated.pop(index_worst_ranking)  # Deleting worst feature from feature list
        # print(ranking_updated)
        # print(features_updated)

    """if len(ranking_updated) == 1:
        use_column[features.index(features_updated[0])] = False
        print(use_column)
        feature_list.append("Without " + features_updated[0])
        print(feature_list)
        scores.append(0)
        print(scores)"""

    collect.collect_graphics.generate_plot(feature_list, scores, "feature_ablation_test")


def score_model1(predictor, X_train, X_test, y_train, y_test, kernel):  # the "identical score everytime" issue is probably here
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    # print(metrics.f1_score(y_test, y_pred))
    # return metrics.f1_score(y_test, y_pred)
    print(metrics.mean_squared_error(y_test, y_pred))
    return metrics.mean_squared_error(y_test, y_pred)








