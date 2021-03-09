'''
Created on Mar 04, 2021

@author: alexandre
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

import collect.collect_graphics


def perform_random_forest_classification(features, output, n_estimators):
    """This method performs the task of random forest classification.

    :param features: a list of features
    :type features: string list
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :type output: string
    :param n_estimators: indicates the number of trees wanted in the classification
    :type n_estimators: int
    """

    model = RandomForestClassifier(n_estimators=n_estimators)

    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS+".csv"), usecols=output)
    Y = df.to_numpy()
        
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES+".csv"), usecols=features)
    X = df.to_numpy()

    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    Y_train = Y_train.ravel()

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    
    rf_probs = model.predict_proba(X_test)[:, 1]
    
    print("F1 score:", metrics.f1_score(Y_test, Y_pred))
    
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    print("Precision:", metrics.precision_score(Y_test, Y_pred))

    print("Recall:", metrics.recall_score(Y_test, Y_pred), "\n")
    
    roc_value = roc_auc_score(Y_test, rf_probs)

    print("roc value:", roc_value)
