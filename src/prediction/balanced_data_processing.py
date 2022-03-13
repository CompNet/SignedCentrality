import random
import numpy as np
import pandas as pd

import os
import path
import consts
import util




def create_balanced_dataset_with_under_sampling_for_classification(csv_folder_path, X, y, features, output):
  """
  In the classification tasks, the use of balanced dataset is necesary, i.e. each class of an output variable is of equal size. 
    Therefore, it creates balanced dataset for binary and ordinal classification tasks.

  :return X_new: The adjusted feature matrix, where each class is represented evenly
  :return y_new: The adjusted output variable, where each class is represented evenly
  """
  
  output_classes = np.unique(y)
  indexes_by_output = {}
  for val in output_classes:
    indexes_by_output[val] = util.which(np.array(y) == val)
  
  class_size = 10000000 # init, something very large
  for val in output_classes:
    size = len(indexes_by_output[val])
    if size < class_size:
      class_size = size
  
  
  indexes_sampled = {}
  for val in output_classes:
    indexes_sampled[val] = random.sample(indexes_by_output[val], class_size)
    
  indexs_new = []
  for val in output_classes:  
    indexs_new.extend(indexes_sampled[val])
  
  y_new = y[indexs_new]
  X_new = X[indexs_new,:]
  
  df_y_new = pd.DataFrame({output:y_new})
  df_y_new.to_csv(os.path.join(csv_folder_path,output+"_balanced.csv"),sep=";",index=False)
  
  df_features = pd.DataFrame(X_new, columns = features)
  df_features.to_csv(os.path.join(csv_folder_path,"features_for_"+output+"_balanced.csv"),sep=";",index=False)
  
  return X_new, y_new



def train_test_split_for_balanced_dataset(csv_folder_path, X, y, features, output):
  """
  It prepared the hyperparameter and train-test datasets for a balanced dataset, in such a way that
    each class of an output variable is represented evenly.

  :return X_new: The adjusted feature matrix, where each class is represented evenly
  :return y_new: The adjusted output variable, where each class is represented evenly
  """
    
  output_classes = np.unique(y)
  indexes_by_output = {}
  for val in output_classes:
    indexes_by_output[val] = util.which(np.array(y) == val)
  
  class_size = len(indexes_by_output[output_classes[0]])
  total_size = class_size * len(output_classes)
  hyperparameter_eval_size = int(class_size*0.3)
  
  indexes_sampled_hyperparameter = {}
  for val in output_classes:
    indexes_sampled_hyperparameter[val] = random.sample(indexes_by_output[val], hyperparameter_eval_size)
    
  indexs_new_hyperparameter = []
  for val in output_classes:  
    indexs_new_hyperparameter.extend(indexes_sampled_hyperparameter[val])
  indexs_new_train_test = list(set(range(total_size)) - set(indexs_new_hyperparameter))
  
  
  y_new_hyperparameter_eval = y[indexs_new_hyperparameter]
  y_new_train_test = y[indexs_new_train_test]
  
  X_new_hyperparameter_eval = X[indexs_new_hyperparameter,:]
  X_new_train_test = X[indexs_new_train_test,:]
  
  df_y_new_hyperparameter_eval = pd.DataFrame({output:y_new_hyperparameter_eval})
  df_y_new_hyperparameter_eval.to_csv(os.path.join(csv_folder_path,output+"_balanced_for_hyperparameter_eval.csv"),sep=";",index=False)
  df_y_new_train_test = pd.DataFrame({output:y_new_train_test})
  df_y_new_train_test.to_csv(os.path.join(csv_folder_path,output+"_balanced_for_train_set_eval.csv"),sep=";",index=False)
  
  df_features_new_hyperparameter_eval = pd.DataFrame(X_new_hyperparameter_eval, columns = features)
  df_features_new_hyperparameter_eval.to_csv(os.path.join(csv_folder_path,"features_for_"+output+"_balanced_for_hyperparameter_eval.csv"),sep=";",index=False)
  df_features_new_train_test = pd.DataFrame(X_new_train_test, columns = features)
  df_features_new_train_test.to_csv(os.path.join(csv_folder_path,"features_for_"+output+"_balanced_for_train_set_eval.csv"),sep=";",index=False)
    




def create_balanced_datasets_with_under_sampling_and_train_split():
  """
  It performs two tasks:
  1) In the classification tasks, the use of balanced dataset is necesary, i.e. each class of an output variable is of equal size. 
    Therefore, it creates balanced dataset for binary and ordinal classification tasks.
  2) It prepared the hyperparameter and train-test datasets for a balanced dataset, in such a way that
    each class of an output variable is represented evenly.

  :return None
  """
    
  csv_folder_path = path.get_csv_folder_path()
  features_filename = consts.FILE_CSV_FEATURES+".csv"
  features_filepath = os.path.join(csv_folder_path,features_filename)
  outputs_filename = consts.FILE_CSV_OUTPUTS_WITH_ORDINAL_VAR+".csv"
  outputs_filepath = os.path.join(csv_folder_path,outputs_filename)

  df_outputs = pd.read_csv(outputs_filepath, sep=",")
  df_features = pd.read_csv(features_filepath, sep=",")
  features = df_features.columns
  X = df_features.to_numpy()
  
  single_solution = df_outputs[consts.OUTPUT_IS_SINGLE_SOLUTION].to_numpy()
  X_new, y_new = create_balanced_dataset_with_under_sampling_for_classification(csv_folder_path, X, single_solution, features, consts.OUTPUT_IS_SINGLE_SOLUTION)
  train_test_split_for_balanced_dataset(csv_folder_path, X_new, y_new, features, consts.OUTPUT_IS_SINGLE_SOLUTION)
  
  single_solution_class = df_outputs[consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES].to_numpy()
  X_new, y_new = create_balanced_dataset_with_under_sampling_for_classification(csv_folder_path, X, single_solution_class, features, consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES)
  train_test_split_for_balanced_dataset(csv_folder_path, X_new, y_new, features, consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES)

  nb_solutions_ordinal = df_outputs[consts.OUTPUT_NB_SOLUTIONS_ORDINAL].to_numpy()
  X_new, y_new = create_balanced_dataset_with_under_sampling_for_classification(csv_folder_path, X, nb_solutions_ordinal, features, consts.OUTPUT_NB_SOLUTIONS_ORDINAL)
  train_test_split_for_balanced_dataset(csv_folder_path, X_new, y_new, features, consts.OUTPUT_NB_SOLUTIONS_ORDINAL)

  nb_solution_classes_ordinal = df_outputs[consts.OUTPUT_NB_SOLUTION_CLASSES_ORDINAL].to_numpy()
  X_new, y_new = create_balanced_dataset_with_under_sampling_for_classification(csv_folder_path, X, nb_solution_classes_ordinal, features, consts.OUTPUT_NB_SOLUTION_CLASSES_ORDINAL)
  train_test_split_for_balanced_dataset(csv_folder_path, X_new, y_new, features, consts.OUTPUT_NB_SOLUTION_CLASSES_ORDINAL)
    

