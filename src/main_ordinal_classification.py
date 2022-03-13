'''
Created on Feb 1, 2022

@author: nejat
'''

import os
import path
import itertools
from time import time

import consts
import pickle

import pandas as pd

from descriptors.runner import compute_all_centralities
import stats.runner
import collect.collect_features
import collect.collect_outputs


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from prediction import ordinal_classification
from prediction import prediction_generic
from prediction import feature_ablation







def perform_ordinal_classification_with_all_models(features, outputs, n_jobs, force=False):
  """
  It performs the ordinal classification task for three classifiers:
  - SVC 
  - Random Forest
  - Multi Layer Perceptron
  Note that we build a generic ordinal classifier (see prediction/ordinal_classification.py),
   which accepts any binary classifier, based on this paper:
  https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf.
  
  It first performs the hyperparameter tuning for each output variable and each classifier, then
  it performs the training and prediction parts. Finally, it performs the feature ablation task.
   
  :param features: feature matrix
  :param outputs: output matrix
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :param force: whether the results are recalculated, although they exist
  :return None
  """
    
  try:
    if not os.path.exists(os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+"ordinal_classification")):
      os.makedirs(os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+"ordinal_classification"))
  except OSError as err:
     print(err)     

  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+"ordinal_classification")):
      os.makedirs(os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+"ordinal_classification"))
  except OSError as err:
     print(err)
     
  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"ordinal_classification")):
      os.makedirs(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"ordinal_classification"))
  except OSError as err:
     print(err) 

  
  # =============================================
  # retrieve features and outputs of interest
  # prepare imblearn class and scoring
  # =============================================

#  imblearn_classes = [RandomOverSampler(), SVMSMOTE()]
  imblearn_classes = [None]
  scorings = ["f1_weighted", "f1_macro", "f1_micro"]
  

  # =============================================
  # prepare ordinal classification models
  # =============================================
    
  # SVC
  ordinal_classifier_model = ordinal_classification.OrdinalClassifier(svm.SVC(probability=True, class_weight="balanced"))
  param_grid = dict( \
                model__kernel=['rbf', 'poly'], \
                #model__kernel=['linear', 'poly', 'rbf', 'sigmoid'], \
                model__C=[0.1, 1, 100], \
                #model__C=[0.1, 1, 10, 25, 50, 75, 100, 150, 200], \
                model__decision_function_shape=['ovo', 'ovr'] \
              )
  svc_ordinal_classif_dict = {"model_name":"svc", "model":ordinal_classifier_model, "param_grid":param_grid}

  # ----------------

  # Random Forest 
  ordinal_classifier_model = ordinal_classification.OrdinalClassifier(RandomForestClassifier(class_weight="balanced"))
  param_grid = dict( \
                  #model__n_estimators=[10, 50, 100, 150], \
                  model__n_estimators=[10, 50], \
                  #model__min_samples_split=[2,4,6,8], \
                  #model__max_depth=[10,50,100,None], \
                  model__max_depth=[10,50,None], \
                  #model__min_samples_leaf=[1,2,3] \
                )
  rf_ordinal_classif_dict = {"model_name":"random_forest", "model":ordinal_classifier_model, "param_grid":param_grid}

  # ----------------
  
  # Multi Layer Perceptron
  ordinal_classifier_model = ordinal_classification.OrdinalClassifier(MLPClassifier())
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
  param_grid = dict( \
                  model__hidden_layer_sizes=layers, \
                  #model__activation=['identity', 'logistic', 'tanh', 'relu'], \
                  model__activation=['logistic', 'tanh'], \
                  model__solver=['adam'], \
                  model__max_iter=[2000], \
                  #model__early_stopping=[False, True] \
                ) 
  mlp_ordinal_classif_dict = {"model_name":"mlpc", "model":ordinal_classifier_model, "param_grid":param_grid}

  # ----------------
  
  ordinal_classifier_dict_list = [svc_ordinal_classif_dict,rf_ordinal_classif_dict,mlp_ordinal_classif_dict]
  # ordinal_classifier_dict_list = [rf_ordinal_classif_dict]

  # =============================================
  # run the models
  # =============================================

  for output in outputs:
    # output_index = util.which(np.array(outputs) == output)[0] # util.which() returns a list of 1 element here
    # Y_hyperparameters_output = Y_hyperparameters[:,output_index]
    # Y_train_test_output = Y_train_test[:,output_index]
    # perform_classification_or_regression_with_generic_model("ordinal_classification", ordinal_classifier_dict_list, X_train_test, X_hyperparameters, Y_train_test_output, Y_hyperparameters_output, output, imblearn_classes, scorings, n_jobs)

    X_train_test, X_hyperparameters, Y_train_test, Y_hyperparameters = prediction_generic.read_balanced_training_and_test_data_for_prediction(output, features)

    prediction_generic.perform_hyperparam_eval_for_classification_or_regression_with_generic_model("ordinal_classification", \
                                                            ordinal_classifier_dict_list, X_hyperparameters, Y_hyperparameters, \
                                                             output, imblearn_classes, scorings, n_jobs, force)
    
    prediction_generic.perform_prediction_for_classification_or_regression_with_generic_model("ordinal_classification", \
                                                            ordinal_classifier_dict_list, X_train_test, Y_train_test, \
                                                             output, imblearn_classes, scorings, n_jobs, force)

    X_train_test2 = pd.DataFrame(X_train_test, columns=features) # convert numpy array into pandas data frame
    feature_ablation.perform_all_feature_ablation("ordinal_classification", ordinal_classifier_dict_list, X_train_test2, Y_train_test, output, scorings, imblearn_classes, force)
  



  
  
  
####################################################################### 
# MAIN
#######################################################################
  

GRAPH_SIZES = [20,24]
L0_VALS = [3] #
PROP_MISPLS = [x/20 for x in range(0, 21)] # float range from 0.0 to 1.0 with decimal steps
DENSITY = [0.25, 0.5, 1] # 
INPUT_NETWORKS = range(1,11)
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
  consts.OUTPUT_NB_SOLUTIONS_ORDINAL, 
  consts.OUTPUT_NB_SOLUTION_CLASSES_ORDINAL
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
  
    perform_ordinal_classification_with_all_models(features, OUTPUTS, N_JOBS_IN_PREDICTION, FORCE)
  
    program_end_time = time() - program_start_time
    print("Running time of the full program:", program_end_time, "seconds")

