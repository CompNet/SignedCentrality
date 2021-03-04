'''
Created on Mar 04, 2021

@author: alexandre
'''

import os
import consts
import path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import collect.collect_graphics


def perform_random_forest_classification(features, output):
    """This method performs the task of random forest classification.

    """
    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100)

    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS+".csv"), usecols=output)
    Y = df.to_numpy()
        
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES+".csv"), usecols=features)
    X = df.to_numpy()
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Fit on training data
    model.fit(X_train, Y_train)

    # Actual class predictions
    Y_pred = model.predict(X_test)
    # Probabilities for each class
    rf_probs = model.predict_proba(Y_test)[:, 1]
    
    print("F1 score:", metrics.f1_score(Y_test, Y_pred))
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, Y_pred)) # "UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples" with the Github reduced dataset

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, Y_pred), "\n")
    
    # Calculate roc auc which indicates the quality of the prediction
    roc_value = roc_auc_score(test_labels, rf_probs)

    print("roc value:", roc_value)
