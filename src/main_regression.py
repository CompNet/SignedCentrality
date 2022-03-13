'''
Created on Feb 1, 2022

@author: nejat
'''

import os
import path
import itertools
from time import time

import consts
import util

from descriptors.runner import compute_all_centralities
import stats.runner
import collect.collect_features
import collect.collect_outputs

import pickle
import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm

from prediction import prediction_generic
from prediction import regression
from prediction import feature_ablation




def plot_regression_results(models_list, output, X_train_test, Y_train_test, scorings, force=False):
  """
  It is a wrapper function for plotting the regression results for a better understanding of the performances of
  the regression models.
  It reads the trained model from a file thanks to the package "pickle".
  - true values vs squared error
  - predicted values vs squared error
  - true values vs predicted values
  
  :param models_list: a list of machine learning models
  :param output: The name of the output variable of interest
  :param X_train_test: Samples for train and test set
  :param Y_train_test: Output variable for train and test set
  :param scorings: Scoring metrics, e.g. "accuracy"
  :param force: whether the results are recalculated, although they exist
  :return None
  """
  
  task_description = "regression"

  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description)):
      os.makedirs(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description))
  except OSError as err:
    print(err) 
  
  for model_dict in models_list:
    model_name = model_dict["model_name"]
    model = model_dict["model"]
    param_grid = model_dict["param_grid"]
    
    for scoring in scorings:
      print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring)
      best_model_pickle_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "best-model="+model_name+"_output="+ output+"_scoring="+scoring+".pkl")

      if os.path.exists(best_model_pickle_filepath) or force:
        # load saved model
        with open(best_model_pickle_filepath , 'rb') as f:
          best_model = pickle.load(f)
          regression.plot_regression_true_pred_error_values(X_train_test, Y_train_test, best_model, model_name, output)




def perform_regression_with_all_models(features, outputs, n_jobs, force=False):
  """
  It performs the regression task for three models:
  - SVR 
  - Linear Regression
  - Multi Layer Perceptron
  
  It first performs the hyperparameter tuning for each output variable and each regressor, then
  it performs the training and prediction parts. Next, it plots the regression results.
  Finally, it performs the feature ablation task.
   
  :param features: feature matrix
  :param outputs: output matrix
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :param force: whether the results are recalculated, although they exist
  :return None
  """
  
  try:
    if not os.path.exists(os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+"regression")):
      os.makedirs(os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+"regression"))
  except OSError as err:
     print(err)     

  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+"regression")):
      os.makedirs(os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+"regression"))
  except OSError as err:
     print(err)
     
  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression")):
      os.makedirs(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression"))
  except OSError as err:
     print(err) 
        
               
  # =============================================
  # retrieve features and outputs of interest
  # prepare imblearn class and scoring
  # =============================================

  X_train_test, X_hyperparameters, Y_train_test, Y_hyperparameters = prediction_generic.prepare_prediction_data(features, outputs, force)
  
  imblearn_classes = [None]
  scorings = ["neg_mean_squared_error"]


  # =============================================
  # prepare regression models
  # =============================================
      
  # SVR
  regression_model = svm.SVR()
  param_grid = dict( \
                model__kernel=['rbf', 'poly'], \
                #model__kernel=['linear', 'poly', 'rbf', 'sigmoid'], \
                model__gamma=['scale', 'auto'], \
              )
  svr_regression_dict = {"model_name":"svr", "model":regression_model, "param_grid":param_grid}

  # ----------------

  # Linear Regression
  regression_model = LinearRegression()
  param_grid = dict( \
                      model__fit_intercept=[False, True], \
                      model__positive=[False, True] \
                    )
  lr_regression_dict = {"model_name":"lr", "model":regression_model, "param_grid":param_grid}

    
  # ----------------
  
  # Multi Layer Perceptron
  regression_model = MLPRegressor()
  layers_numbers = [1, 3]
  #layer_sizes = [1, 2, 3, 4, 5, 10]
  layer_sizes = [1, 5]
  # layer_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
  layers = []
  for layers_number in layers_numbers:
    for layer_size in layer_sizes:
      # layer no = 3
      # layer size = 5
      layers.append(tuple([layer_size]*layers_number))
  #classifier_model.set_imblearn_class(imblearn_class)
  param_grid = dict( \
                  model__hidden_layer_sizes=layers, \
                  #model__activation=['identity', 'logistic', 'tanh', 'relu'], \
                  model__activation=['logistic', 'tanh'], \
                  model__solver=['adam'], \
                  model__max_iter=[2000], \
                  #model__early_stopping=[False, True] \
                ) 
  mlp_regression_dict = {"model_name":"mlpr", "model":regression_model, "param_grid":param_grid}
  
  # -----------

  regressor_dict_list = [svr_regression_dict,lr_regression_dict,mlp_regression_dict]

  # =============================================
  # run the models
  # =============================================
  
  for output in outputs:
    output_index = util.which(np.array(outputs) == output)[0] # util.which() returns a list of 1 element here
    Y_hyperparameters_output = Y_hyperparameters[:,output_index]
    Y_train_test_output = Y_train_test[:,output_index]
    
    prediction_generic.perform_hyperparam_eval_for_classification_or_regression_with_generic_model("regression", \
                                                            regressor_dict_list, X_hyperparameters, Y_hyperparameters_output, \
                                                             output, imblearn_classes, scorings, n_jobs, force)
    
    prediction_generic.perform_prediction_for_classification_or_regression_with_generic_model("regression", \
                                                            regressor_dict_list, X_train_test, Y_train_test_output, \
                                                             output, imblearn_classes, scorings, n_jobs, force)

    plot_regression_results(regressor_dict_list, output, X_train_test, Y_train_test_output, scorings, force)

    X_train_test2 = pd.DataFrame(X_train_test, columns=features) # convert numpy array into pandas data frame
    feature_ablation.perform_all_feature_ablation("regression", regressor_dict_list, X_train_test2, Y_train_test_output, output, scorings, imblearn_classes, force)

  
    
    
    
    
    
####################################################################### 
# MAIN
#######################################################################

GRAPH_SIZES = [20,24,28]
L0_VALS = [2,3,4] #
PROP_MISPLS = [x/20 for x in range(0, 21)] # float range from 0.0 to 1.0 with decimal steps
DENSITY = [0.25, 0.5, 1] # 
INPUT_NETWORKS = range(1,21)
PROP_NEGS = [0.3, 0.5, 0.7] # when density=1, this equals 'None'

NETWORK_DESC = consts.SIGNED_UNWEIGHTED

GRAPH_DESCRIPTORS = [
  'PNCentrality', 
  'EigenvectorCentrality', 
  'diversity_coef_centrality', 
  'SNEEmbedding', 
  'SiNEEmbedding', 
  'StEMEmbedding'
]

STATS = [
  consts.STATS_NB_NODES,
  consts.STATS_POS_PROP, 
  consts.STATS_NEG_PROP,
  consts.STATS_POS_NEG_RATIO, 
  consts.STATS_SIGNED_TRIANGLES, 
  consts.STATS_LARGEST_EIGENVALUE
]

OUTPUTS = [
  consts.OUTPUT_NB_SOLUTIONS,
  consts.OUTPUT_NB_SOLUTION_CLASSES,
  ##consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE
]

FORCE = False
#FORCE = True
VERBOSE = True

N_JOBS_IN_PREDICTION = 6  


if __name__ == '__main__':

  program_start_time = time()
  
  features = []
  for stat in STATS:
    features.extend(consts.COL_NAMES[stat])

  for descriptor in GRAPH_DESCRIPTORS:
    features.extend([
        consts.PREFIX_MEAN + descriptor,
        consts.PREFIX_STD + descriptor
    ])

  perform_regression_with_all_models(features, OUTPUTS, N_JOBS_IN_PREDICTION, FORCE)

  program_end_time = time() - program_start_time
  print("Running time of the full program:", program_end_time, "seconds")

