'''
Created on Feb 24, 2021

@author: Laurent PEREIRA DA SILVA
'''


import util
import consts
import path
import os
from statistics import mean, stdev

import pandas as pd
import numpy as np


def collect_predicted_values(model_descr, true_values, predicted_values, prediction_name, verbose=False):
    """
    Write predicted values in a CSV file

    Initial version, where only the predicted values are saved to file, by "brute force".

    :param predicted_values: Values to write in a file
    :param prediction_name: Name of the prediction technique
    :param verbose: True if information must be printed
    """
    print(true_values.shape)
    print(predicted_values.shape)
    dataset = pd.DataFrame({'Y_true': true_values.flatten(), 'Y_pred': predicted_values.flatten()})

    # preparations of multiples variables
    prediction_name = str(prediction_name)
    prediction_name = util.prediction_name_refactor(prediction_name)
    csv_folder_path = path.get_csv_folder_path()
    os.makedirs(csv_folder_path, exist_ok=True)
    result_filename = prediction_name + "_" + model_descr + "_predicted_values.csv"
    result_filepath = os.path.join(csv_folder_path, result_filename)
    # print(predicted_values)
    """for x in np.nditer(predicted_values): # method to select all values inside the ndarray
        print(type(x), end=' ')
        print(x, end=' ')
        x = float(x)
        print(type(x), end=' ')
        print(x, end=' ')"""

    if verbose:
        print("Collecting predicted data for " + prediction_name)
    if not os.path.exists(result_filepath):
        # for x in np.nditer(predicted_values):  # method to select all values inside the ndarray : https://numpy.org/doc/stable/reference/arrays.nditer.html
        #     result = None
        #     result = [x]
        #     print(result)
        #     df = pd.DataFrame(data=result, index=[prediction_name]).transpose()
        #     predicted_dataset = predicted_dataset.append(df)
        #     print(predicted_dataset)
        # # predicted_dataset.append(predicted_values)
        dataset.to_csv(result_filepath, sep=",", quoting=1, index=True)
    else:
        if verbose:
            print(result_filepath + " already exists", "\n")


def collect_predicted_values_with_graph_name(predicted_values, prediction_name, verbose=False):
    """
    Write predicted values in a CSV file

    Updated version, with the predicted values and the graph name associated.
    first, collect all graph names (from outputs.csv file for example)
    then, write it in csv the "same way" as "collect_all_outputs" method (at least for the graph name)
    after that, collect the predicted dataset and extract elements for it, and write it row by row, next to the graph name
    convert everything to csv file at the end

    :param predicted_values: Values to write in a file
    :param prediction_name: Name of the prediction technique
    :param verbose: True if information must be printed
    """

    # graph_names = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS+"_full.csv"), usecols=[0])
    graph_names = pd.read_csv(os.path.join(path.get_csv_folder_path(), consts.FILE_CSV_OUTPUTS + ".csv"), usecols=[0])
    # print(graph_names)
    predicted_dataset = pd.DataFrame([])

    # preparations of multiples variables
    prediction_name = str(prediction_name)
    prediction_name = util.prediction_name_refactor(prediction_name)
    csv_folder_path = path.get_csv_folder_path()
    os.makedirs(csv_folder_path, exist_ok=True)
    result_filename = prediction_name + "_predicted_values.csv"
    result_filepath = os.path.join(csv_folder_path, result_filename)
    # print(predicted_values)
    """for x in np.nditer(predicted_values): # method to select all values inside the ndarray : https://numpy.org/doc/stable/reference/arrays.nditer.html
        print(type(x), end=' ')
        print(x, end=' ')
        x = float(x)
        print(type(x), end=' ')
        print(x, end=' ')
    """

    print("Collecting predicted data for " + prediction_name)
    if not os.path.exists(result_filepath):
        for x in np.nditer(predicted_values):  # method to select all values inside the ndarray : https://numpy.org/doc/stable/reference/arrays.nditer.html
            result = None
            result = [x]
            # print(result)
            df = pd.DataFrame(data=result, index=[prediction_name]).transpose()
            predicted_dataset = predicted_dataset.append(df)
            # print(predicted_dataset)
        # predicted_dataset.append(predicted_values)
        # predicted_dataset.index = graph_names # too many graph names, because of the dataset split for prediction
        predicted_dataset.to_csv(result_filepath, sep=",", quoting=1, index=True)
    else:
        if verbose:
            print(result_filepath + " already exists", "\n")


