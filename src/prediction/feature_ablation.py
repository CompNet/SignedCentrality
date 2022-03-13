#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re

# Import train_test_split function
# Import svm model
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import EditedNearestNeighbours

import collect.collect_graphics
from collect import collect_graphics
from prediction import initialize_hyper_parameters, initialize_data

import pickle

# ===========================================================================================
# TODO: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# https://scikit-learn.org/stable/modules/feature_selection.html#rfe
# ===========================================================================================






def get_score_after_permutation(model, X, y, curr_feat):
  """ return the score of model when curr_feat is permuted """

  X_permuted = X.copy()
  col_idx = list(X.columns).index(curr_feat)
  # permute one column
  X_permuted.iloc[:, col_idx] = np.random.permutation(X_permuted[curr_feat].values)

  permuted_score = model.score(X_permuted, y)
  return permuted_score



def get_feature_importance(model, X, y, curr_feat):
  """ compare the score when curr_feat is permuted """

  baseline_score_train = model.score(X, y)
  permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

  # feature importance is the difference between the two scores
  feature_importance = baseline_score_train - permuted_score_train
  return feature_importance


def permutation_importance(model, X, y, n_repeats=10):
  """Calculate importance score for each feature."""

  importances = []
  for curr_feat in X.columns:
    list_feature_importance = []
    for n_round in range(n_repeats):
      list_feature_importance.append(get_feature_importance(model, X, y, curr_feat))
    importances.append(list_feature_importance)

  return {'importances_mean': np.mean(importances, axis=1), 'importances_std': np.std(importances, axis=1), 'importances': importances}



def plot_important_features(perm_importance_result, feat_name, plot_filepath):
  """ bar plot the feature importance """

  fig, ax = plt.subplots()

  indices = perm_importance_result['importances_mean'].argsort()
  plt.barh(range(len(indices)), perm_importance_result['importances_mean'][indices], xerr=perm_importance_result['importances_std'][indices])

  ax.set_yticks(range(len(indices)))
  _ = ax.set_yticklabels(feat_name[indices])
  
  plt.savefig(plot_filepath, format='pdf')

# This function could directly be access from sklearn
# from sklearn.inspection import permutation_importance
  
  
  



def perform_feature_ablation(task_description, X_train_test, Y_train_test, model, model_name, output, scoring, imblearn_class, force):
  """
  It performs the feature ablation task. The source code is taken from this website:
  https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html

  :param task_description: task description, e.g. "binary_classification"
  :param models_list: a list of machine learning models
  :param X_train_test: Samples for train and test set
  :param Y_train_test: Output variable for train and test set
  :param output: The name of the output variable of interest
  :param scorings: Scoring metrics
  :param imblearn_classes (deprecated): Classes dealing with imbalanced data.
  :param force: whether the results are recalculated, although they exist
  :return None
  """
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test, test_size=0.3, random_state=0)
  #model = reg.fit(X_train, Y_train)
  #score = model.score(X_test, Y_test) # r2
  #print(score)
  # nb_test_samples = Y_test.shape[0]
  # print(nb_test_samples)
  # print("----")
  # print(Y_test[5])
  # print(model.predict(X_test[5,].reshape(1, -1)))
  
  imblearn_class_descr = "None"
  if imblearn_class is not None:
    imblearn_class_descr = imblearn_class.__str__()
  
  plot_filepath = os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description, "feature-ablation_model="+model_name+"_output="+ output+"_scoring="+scoring+".pdf")
  if not task_description.startswith("regression"):
    plot_filepath = os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description, "feature-ablation_model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".pdf")
  
  if not os.path.exists(plot_filepath) or force:
    perm_importance_result_train = permutation_importance(model, X_train, Y_train, n_repeats=10)
  
    plot_important_features(perm_importance_result_train, X_train.columns, plot_filepath)
    
    


    
def perform_all_feature_ablation(task_description, models_list, X_train_test, Y_train_test, output, scorings, imblearn_classes, force):
  """
  It is a wrapper function to perform the feature ablation task. It iterates over:
  - different machine learning models
  - scoring metrics
  - classes dealing with imbalanced data

  :param task_description: task description, e.g. "binary_classification"
  :param models_list: a list of machine learning models
  :param X_train_test: Samples for train and test set
  :param Y_train_test: Output variable for train and test set
  :param output: The name of the output variable of interest
  :param scorings: Scoring metrics
  :param imblearn_classes (deprecated): Classes dealing with imbalanced data.
  :param force: whether the results are recalculated, although they exist
  :return None
  """

  try:
    if not os.path.exists(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description)):
      os.makedirs(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+task_description))
  except OSError as err:
    print(err)     
  

  for imblearn_class in imblearn_classes:
    imblearn_class_descr = "None"
    if imblearn_class is not None:
      imblearn_class_descr = imblearn_class.__str__()
    
    for model_dict in models_list:
      model_name = model_dict["model_name"]
      model = model_dict["model"]
      param_grid = model_dict["param_grid"]
      
      for scoring in scorings:
        print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring)
        
        best_model_pickle_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "best-model="+model_name+"_output="+ output+"_scoring="+scoring+".pkl")
        if not task_description.startswith("regression"):
          best_model_pickle_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "best-model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".pkl")

        if os.path.exists(best_model_pickle_filepath) or force:
          # load saved model
          with open(best_model_pickle_filepath , 'rb') as f:
            best_model = pickle.load(f)
            
            perform_feature_ablation(task_description, X_train_test, Y_train_test, best_model, model_name, output, scoring, imblearn_class, force)



