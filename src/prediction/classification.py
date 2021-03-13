'''
Created on Sep 23, 2020

@author: nejat
'''

import os
import consts
import path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import collect.collect_graphics


# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/metrics.html#linear-kernel
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#clustering-metrics

# TODO: 
# - k fold test
def perform_classification(features, output, kernel):
    """This method performs the task of classification for a single output.
       
    :param features: a list of features
    :type features: string list
    :param output: a single output, e.g. consts.OUTPUT_NB_SOLUTIONS
    :type output: string
    :param kernel: a kernel model, e.g. consts.PREDICTION_KERNEL_LINEAR, etc.
    :type kernel: string
    """
    # =======================================================
    # Read features and output from file (original code)
    # =======================================================
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS+".csv"), usecols=output)
    Y = df.to_numpy()
    
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES+".csv"), usecols=features)
    X = df.to_numpy()

    # =======================================================
    # Read features and output from file (test code)
    # =======================================================
    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_full.csv"), usecols=output)
    # Y = df.to_numpy()

    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_full.csv"), usecols=features)
    # X = df.to_numpy()
    
    scaler = StandardScaler()
    #scaler.fit(X[:,0].reshape(-1,1))
    #X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)
    
    # =======================================================
    # Split data intro train and test sets
    # =======================================================
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109) # 70% training and 30% test
    Y_train = Y_train.ravel() # convert into 1D array, due to the warning from 'train_test_split'
    
    # =======================================================
    #  Train: Create a svm Classifier
    # =======================================================
    # >> other params: gamma, max_iter, degree, decision_function_shape, shrinking
    # clf = svm.SVC(kernel=kernel) # original code
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    
    # =======================================================
    # Test: Predict the response for test dataset
    # =======================================================
    Y_pred = clf.predict(X_test) # Returns a numpy.ndarray

    
    # =======================================================
    # Metrics
    # =======================================================
    # print("F1 score:", metrics.f1_score(Y_test, Y_pred),"\n")
    # print("Test dataset:",Y_test)
    # print("Predicted dataset:", Y_pred)
    print("F1 score:", metrics.f1_score(Y_test, Y_pred))
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, Y_pred)) # "UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples" with the Github reduced dataset

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, Y_pred), "\n")

    # Model Precision & Recall
    # print("Precision & Recall:", metrics.precision_recall_fscore_support(Y_test, Y_pred))
    
    # Regression metrics for training : mean_squared_error
    #print("mean squared error:", metrics.mean_squared_error(Y_test, Y_pred))

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)

    