'''
Created on Feb 24, 2021

@author: Laurent PEREIRA DA SILVA
'''


import util
import centrality.degree_centrality
import centrality.eigenvector_centrality
import consts
import path
import os
from statistics import mean, stdev

import pandas as pd

def collect_predicted_values(predicted_values, prediction_name):
    # first, collect all graph names (from outputs.csv file for example)
    # then, write it in csv the "same way" as "collect_all_outputs" method (at least for the graph name)
    # after that, collect the predicted dataset and extract elements for it, and write it row by row, next to the graph name
    # convert everything to csv file at the end

    predicted_dataset = pd.DataFrame([])
    csv_folder_path = path.get_csv_folder_path()
    os.makedirs(csv_folder_path, exist_ok=True)
    result_filename = prediction_name + ".csv"
    result_filepath = os.path.join(csv_folder_path, result_filename)

    if not os.path.exists(result_filepath):
        print("work in progress")
    else:
        print(result_filepath + " already exists")

