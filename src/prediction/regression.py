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
import collect.collect_predicted_values

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# https://scikit-learn.org/stable/modules/metrics.html#linear-kernel
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

def perform_regression(features, output, kernel):
    """This method performs the task of regression for a single output.

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
    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=output)
    Y = df.to_numpy()

    df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + ".csv"), usecols=features)
    X = df.to_numpy()

    # =======================================================
    # Read features and output from file (test code)
    # =======================================================
    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + "_full.csv"), usecols=output)
    # Y = df.to_numpy()

    # df = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_FEATURES + "_full.csv"), usecols=features)
    # X = df.to_numpy()

    scaler = StandardScaler()
    # scaler.fit(X[:,0].reshape(-1,1))
    # X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1)
    scaler.fit(X)
    X = scaler.transform(X)

    # =======================================================
    # Split data intro train and test sets
    # =======================================================
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    Y_train = Y_train.ravel()  # convert into 1D array, due to the warning from 'train_test_split'

    # =======================================================
    #  Train: Create a svm Regressor
    # =======================================================
    # >> other params: gamma, max_iter, degree, shrinking
    # reg = svm.SVC(kernel=kernel) # original code
    reg = svm.SVR(kernel='linear')
    reg.fit(X_train, Y_train)

    # =======================================================
    # Test: Predict the response for test dataset
    # =======================================================
    Y_pred = reg.predict(X_test) # Returns a numpy.ndarray


    # print("Predicted dataset before rounding:", Y_pred) # I want to transform decimal values to integer values
    # i = 0
    # for val in Y_pred:
    #     print("Value before rounding:", val)
    #     Y_pred[i] = round(val)
    #     print("Value after rounding:",Y_pred[i])
    #     i += 1
    # print("Predicted dataset after rounding:", Y_pred)

    # =======================================================
    # Metrics
    # =======================================================
    # print("Test dataset:", Y_test)
    # print("Predicted dataset:", Y_pred)
    print("R2 score:", metrics.r2_score(Y_test, Y_pred))


    print("Mean squared error:", metrics.mean_squared_error(Y_test, Y_pred),"\n")

    # Saving predicted values to file
    collect.collect_predicted_values.collect_predicted_values(Y_pred, output)

    # Saving graphics to file
    # collect.collect_graphics.generate_plot(Y_test, Y_pred, output)
    # collect.collect_graphics.generate_boxplot(Y_test, Y_pred, output)

