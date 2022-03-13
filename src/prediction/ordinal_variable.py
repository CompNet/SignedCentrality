'''
Created on Feb 1, 2022

@author: nejat
'''

import os
import numpy as np
import pandas as pd
import path
import consts
import util


def create_ordinal_output_variable_for_number_of_optimal_solutions(nb_solutions):
  
    classes_for_nb_solutions = ["-1"]*len(nb_solutions)
    for idx in util.which(nb_solutions == 1):
        classes_for_nb_solutions[idx] = "1"
    for idx in util.which((nb_solutions >= 2) & (nb_solutions <= 5)):
        classes_for_nb_solutions[idx] = "(1,5]"
    for idx in util.which((nb_solutions >= 6) & (nb_solutions <= 25)):
        classes_for_nb_solutions[idx] = "(5,25]"
    for idx in util.which(nb_solutions >= 26):
        classes_for_nb_solutions[idx] = "(25,10000]"
    
    # -----------------------------------------------------------------------
    # This code can be an alternative way of doing this discretization in an automatic manner:
    #
    # classes_for_nb_solutions = np.zeros(len(nb_solutions), dtype='U20')
    # print(classes_for_nb_solutions)
    # single_sols_indexes = np.array(util.which(nb_solutions == 1))
    # multiple_sols_indexes = np.array(util.which(nb_solutions > 1))
    # nb_solutions2 = nb_solutions[multiple_sols_indexes]
    #
    # res = pd.qcut(nb_solutions2, 3)
    # res2 = ["("+str(int(np.ceil(interval.left)))+","+str(int(np.round(interval.right)))+"]" for interval in res]
    # classes_for_nb_solutions[single_sols_indexes] = "1"
    # classes_for_nb_solutions[multiple_sols_indexes] = res2
    # -----------------------------------------------------------------------
    
    return classes_for_nb_solutions
  
  
  
  
def create_ordinal_output_variable_for_number_of_solution_classes(nb_solution_classes):
  
    classes_for_nb_solution_classes = np.zeros(len(nb_solution_classes), dtype='U20')
    
    for idx in util.which(nb_solution_classes == 1):
        classes_for_nb_solution_classes[idx] = "1"
    for idx in util.which(nb_solution_classes == 2):
        classes_for_nb_solution_classes[idx] = "2"
    for idx in util.which(nb_solution_classes == 3):
        classes_for_nb_solution_classes[idx] = "3"
    for idx in util.which((nb_solution_classes >= 4) & (nb_solution_classes <= 25)):
        classes_for_nb_solution_classes[idx] = "(3,25]"
    
    return classes_for_nb_solution_classes





def create_ordinal_output_variables():
  
    csv_folder_path = path.get_csv_folder_path()
    result_filename = consts.FILE_CSV_OUTPUTS+".csv"
    result_filepath = os.path.join(csv_folder_path,result_filename)

    df = pd.read_csv(result_filepath, sep=",")
    nb_solutions = df[consts.OUTPUT_NB_SOLUTIONS].to_numpy()
    nb_solution_classes = df[consts.OUTPUT_NB_SOLUTION_CLASSES].to_numpy()
    
    classes_for_nb_solutions = create_ordinal_output_variable_for_number_of_optimal_solutions(nb_solutions)
    df[consts.OUTPUT_NB_SOLUTIONS_ORDINAL] = classes_for_nb_solutions  

    classes_for_nb_solution_classes = create_ordinal_output_variable_for_number_of_solution_classes(nb_solution_classes)
    df[consts.OUTPUT_NB_SOLUTION_CLASSES_ORDINAL] = classes_for_nb_solution_classes

    new_result_filepath = os.path.join(csv_folder_path, consts.FILE_CSV_OUTPUTS_WITH_ORDINAL_VAR+".csv")
    print(new_result_filepath)
    df.to_csv(new_result_filepath, sep=",", index=False)






