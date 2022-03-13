'''
Created on Feb 1, 2022

@author: nejat
'''


import os
import consts
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

from prediction import regression

import pickle



# Remark: we use the parameter "random_state" for reproducibility purposes.
#  Nevertheless, we can omit this parameter, since this function handles if the csv files are already present or not
def prepare_prediction_data(features=None, outputs=None, force=False):
  """
  It splits the features and output datasets into two parts: 1) evaluation for hyperparameter tuning, 2) train/test part.
  This function does not pay attention if the dataset has balanced classes or not.
  Therefore, it is only used for the regression task.

  :param features: Feature matrix
  :param outputs: Output matrix
  :param force: whether the results are recalculated, although they exist
  :return X_train_test
  :return X_hyperparameters
  :return Y_train_test
  :return Y_hyperparameters
  """
    
  X_train_test_filepath = os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_FEATURES + "_train_test" + consts.CSV)
  Y_train_test_filepath = os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_OUTPUTS + "_train_test" + consts.CSV)
  X_hyperparameters_filepath = os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_FEATURES + "_hyperparameters" + consts.CSV)
  Y_hyperparameters_filepath = os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_OUTPUTS + "_hyperparameters" + consts.CSV)

  if (not (os.path.exists(X_train_test_filepath) or os.path.exists(Y_train_test_filepath) \
      or os.path.exists(X_hyperparameters_filepath) or os.path.exists(Y_hyperparameters_filepath))) or force:
    print("X and Y variables are not split")

    df = pd.read_csv(os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_OUTPUTS_WITH_ORDINAL_VAR + consts.CSV)) # read all outputs
    if outputs is not None:
      df = pd.read_csv(os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_OUTPUTS_WITH_ORDINAL_VAR + consts.CSV), usecols=outputs)
    outputs = df.columns # read the column names
    Y = df.to_numpy()
    
    df = pd.read_csv(os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_FEATURES + consts.CSV)) # read all features
    if features is not None:
      df = pd.read_csv(os.path.join(consts.CSV_FOLDER, consts.FILE_CSV_FEATURES + consts.CSV), usecols=features)
    features = df.columns # read the column names
    X = df.to_numpy()
    
    X_train_test, X_hyperparameters, Y_train_test, Y_hyperparameters = train_test_split(X, Y, test_size=0.3, random_state=0)
    df_X_train_test = pd.DataFrame(X_train_test, columns = features)
    df_X_train_test.to_csv(X_train_test_filepath,sep=",",index=False)
    df_Y_train_test = pd.DataFrame(Y_train_test, columns = outputs) 
    df_Y_train_test.to_csv(Y_train_test_filepath,sep=",",index=False)
    df_X_hyperparameters = pd.DataFrame(X_hyperparameters, columns = features)
    df_X_hyperparameters.to_csv(X_hyperparameters_filepath,sep=",",index=False)
    df_Y_hyperparameters = pd.DataFrame(Y_hyperparameters, columns = outputs)
    df_Y_hyperparameters.to_csv(Y_hyperparameters_filepath,sep=",",index=False)
    
  else:
    print("X and Y variables are already split")

    X_train_test = pd.read_csv(X_train_test_filepath).to_numpy()
    X_hyperparameters = pd.read_csv(X_hyperparameters_filepath).to_numpy()
    if features is not None:
      X_train_test = pd.read_csv(X_train_test_filepath, usecols=features).to_numpy()
      X_hyperparameters = pd.read_csv(X_hyperparameters_filepath, usecols=features).to_numpy()

    Y_train_test = pd.read_csv(Y_train_test_filepath).to_numpy()
    Y_hyperparameters = pd.read_csv(Y_hyperparameters_filepath).to_numpy()
    if outputs is not None:
      print(Y_train_test_filepath)
      Y_train_test = pd.read_csv(Y_train_test_filepath, usecols=outputs).to_numpy()
      Y_hyperparameters = pd.read_csv(Y_hyperparameters_filepath, usecols=outputs).to_numpy()
  
  #print(X_train_test.shape)
  #print(X_hyperparameters.shape)
  return X_train_test, X_hyperparameters, Y_train_test, Y_hyperparameters





def read_balanced_training_and_test_data_for_prediction(output, features=None):
  """
  Instead of splitting the features and output datasets, it reads the datasets which are already processed by
  the functions "create_balanced_dataset_with_under_sampling_for_classification()" and 
  "train_test_split_for_balanced_dataset()".
  It is used for the classification tasks.

  :param features: Feature matrix
  :param outputs: Output matrix
  :return X_train_test
  :return X_hyperparameters
  :return Y_train_test
  :return Y_hyperparameters
  """
  
      
  X_hyperparameters_filepath = os.path.join(consts.CSV_FOLDER, "features_for_"+output+"_balanced_for_hyperparameter_eval" + consts.CSV)
  Y_hyperparameters_filepath = os.path.join(consts.CSV_FOLDER, output+"_balanced_for_hyperparameter_eval" + consts.CSV)
  X_train_test_filepath = os.path.join(consts.CSV_FOLDER, "features_for_"+output+"_balanced_for_train_set_eval" + consts.CSV)
  Y_train_test_filepath = os.path.join(consts.CSV_FOLDER, output+"_balanced_for_train_set_eval" + consts.CSV)


  X_train_test = pd.read_csv(X_train_test_filepath, sep=";").to_numpy()
  X_hyperparameters = pd.read_csv(X_hyperparameters_filepath, sep=";").to_numpy()
  if features is not None:
    X_train_test = pd.read_csv(X_train_test_filepath, usecols=features, sep=";").to_numpy()
    X_hyperparameters = pd.read_csv(X_hyperparameters_filepath, usecols=features, sep=";").to_numpy()

  Y_train_test = pd.read_csv(Y_train_test_filepath, sep=";").to_numpy()
  Y_hyperparameters = pd.read_csv(Y_hyperparameters_filepath, sep=";").to_numpy()
  
  print(X_train_test.shape)
  print(X_hyperparameters.shape)
  return X_train_test, X_hyperparameters, Y_train_test, Y_hyperparameters








# Tuning the hyper-parameters of an estimator
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search

# Metrics and scoring: quantifying the quality of predictions
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

# Use of hyper param tuning and feature selection (the orders)
# https://stats.stackexchange.com/a/323899

# pipeline
# https://scikit-learn.org/stable/modules/compose.html#pipeline


def perform_hyperparameters_eval_for_classification_or_regression(task_description, X, Y, model, param_grid, output_filepath, counts_filepath, imblearn_class, scoring, n_jobs):
  """
  It is a generic function, which deals with the hyperparameter tuning.
  It is used for both the classification and regression tasks.
  It performs three tasks:
  - preprocessing
  - Sckit-learn pipeline with grid search
  - it records the results into a file

  :param task_description: task description, e.g. "binary_classification"
  :param X: Samples for train and test set
  :param Y: Output variable for train and test set
  :param model: a list of machine learning models
  :param param_grid: The name of the output variable of interest
  :param output_filepath: Scoring metrics
  :param counts_filepath (deprecated): Related to the class sizes of an output variable.
  :param imblearn_class (deprecated): Class dealing with imbalanced data.
  :param scoring: Scoring metric, e.g. "accuracy"
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :return None
  """
  
  # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
  
  # =============================================================
  # Preprocessing
  # =============================================================
  Y = Y.ravel() # convert into 1D array, due to the warning

  if not task_description.startswith("regression"):
    unique, counts = np.unique(Y, return_counts=True)
    unique = ["before, " + str(uint) for uint in unique]
    d_before = dict(zip(unique, counts))
  
    # ----------------------------------------
    # make the dataset balanced >> Deprecated. "imblearn_class" is supposed to be 'None'
    if imblearn_class is not None:
      X, Y = imblearn_class.fit_resample(X, Y)
  
    unique, counts = np.unique(Y, return_counts=True)
    unique = ["after, " + str(uint) for uint in unique]
    d_after = dict(zip(unique, counts))
    d_before.update(d_after)
    
    df_counts = pd.DataFrame([d_before])
    df_counts.to_csv(counts_filepath, sep=";", index=False)
    # ----------------------------------------

  # Why we scale the data ? The answer: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)

  
  # =================================================================
  # Sckit-learn pipeline with grid search
  # =================================================================
  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  
  
  # Feature selection is usually used as a pre-processing step before doing the actual learning
  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
  # estimators = [('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False))), ('classif', svm.SVC())]
  # estimators = [('feature_selection', SequentialFeatureSelector(svm.SVC(), direction="forward")), ('classif', svm.SVC())]
  estimators = [('model', model)]
  pipe = Pipeline(estimators)
  
  # cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
  cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
  
  grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs = n_jobs)
  grid_search.fit(X, Y)
  #print(grid_search.cv_results_)
  results_df = pd.DataFrame(grid_search.cv_results_)
  results_df = results_df.sort_values(by=["rank_test_score"])
  results_df = results_df.set_index(
      results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
  ).rename_axis("kernel")
  # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
  #print(results_df[["params", "rank_test_score", "mean_test_score"]])
  #print(grid_search.best_estimator_.named_steps["feature_selection"].get_feature_names_out(input_features=features))
  #print(grid_search.best_estimator_.named_steps["feature_selection"].get_support(indices=True))
  df = results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

  # Why we negate ? The answer: https://stackoverflow.com/questions/48244219/is-sklearn-metrics-mean-squared-error-the-larger-the-better-negated
  if task_description == "regression" and scoring == "neg_mean_squared_error":
    df["mean_test_score"] = - df["mean_test_score"]
    
  df.to_csv(output_filepath, sep=";", index=False)





# Metrics and scoring: quantifying the quality of predictions
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

# pipeline
# https://scikit-learn.org/stable/modules/compose.html#pipeline

def perform_classification_or_regression_prediction(task_description, X, Y, model, output_filepath, counts_filepath, hyperparameters_result_filepath, imblearn_class, scoring, n_jobs):
  """
  It is a generic function, which deals with the training and prediction parts.
  It is used for both the classification and regression tasks.
  It is dependent of the function "perform_hyperparameters_eval_for_classification_or_regression()",
  because it uses the hyperparameters giving the best score.
  It performs three tasks:
  - preprocessing
  - Sckit-learn pipeline with grid search and RepeatedKFold
  - it records the results into a file
  
  Remark: Normally, we do not have to use the grid search mechanism, but it is a workaround for using custom
  classifiers, such as our ordinal classifier. This is why we keep it in this fuction.

  :param task_description: task description, e.g. "binary_classification"
  :param X: Samples for train and test set
  :param Y: Output variable for train and test set
  :param model: a list of machine learning models
  :param param_grid: The name of the output variable of interest
  :param output_filepath: Scoring metrics
  :param counts_filepath (deprecated): Related to the class sizes of an output variable.
  :param imblearn_class (deprecated): Class dealing with imbalanced data.
  :param scoring: Scoring metric, e.g. "accuracy"
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :return None
  """
  
    
  # =============================================================
  # Preprocessing
  # =============================================================
  Y = Y.ravel() # convert into 1D array, due to the warning
  
  if not task_description.startswith("regression"):
    unique, counts = np.unique(Y, return_counts=True)
    unique = ["before, " + str(uint) for uint in unique]
    d_before = dict(zip(unique, counts))
  
    # ----------------------------------------
    # make the dataset balanced >> Deprecated. "imblearn_class" is supposed to be 'None'    if imblearn_class is not None:
    if imblearn_class is not None:
      X, Y = imblearn_class.fit_resample(X, Y)
  
    unique, counts = np.unique(Y, return_counts=True)
    unique = ["after, " + str(uint) for uint in unique]
    d_after = dict(zip(unique, counts))
    d_before.update(d_after)
    
    df_counts = pd.DataFrame([d_before])
    df_counts.to_csv(counts_filepath, sep=";", index=False)
    # ----------------------------------------
    
  # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)

  
  # =================================================================
  # Sckit-learn pipeline with RepeatedKFold
  # =================================================================
  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  
  params_dict = eval(pd.read_csv(hyperparameters_result_filepath, sep=";", usecols=["params"]).iloc[1].loc["params"])
  for key in params_dict.keys():
    params_dict[key] = [params_dict[key]]
  
  estimators = [('model', model)]
  pipe = Pipeline(estimators)
  
  # cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
  cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
  
  grid_search = GridSearchCV(pipe, param_grid=params_dict, scoring=scoring, cv=cv, n_jobs = n_jobs)
  grid_search.fit(X, Y)
  results_df = pd.DataFrame(grid_search.cv_results_)
  #results_df = results_df.sort_values(by=["rank_test_score"])
  results_df2 = results_df[["params","mean_test_score","std_test_score"]]
  results_df2["scoring"] = scoring
  
  # https://stackoverflow.com/questions/48244219/is-sklearn-metrics-mean-squared-error-the-larger-the-better-negated
  if task_description.startswith("regression") and scoring == "neg_mean_squared_error":
    results_df2["mean_test_score"] = - results_df2["mean_test_score"]
    
  results_df2.to_csv(output_filepath, sep=";", index=False)
  return grid_search.best_estimator_
 
 
 
def perform_hyperparam_eval_for_classification_or_regression_with_generic_model(task_description, models_list, \
                                      X_hyperparameters, Y_hyperparameters,\
                                       output, imblearn_classes, scorings, n_jobs, force=False):
  """
  It is a wrapper function to perform the hyperparameter tuning. It iterates over:
  - different machine learning models
  - scoring metrics
  - classes dealing with imbalanced data

  :param task_description: task description, e.g. "binary_classification"
  :param models_list: a list of machine learning models
  :param X_hyperparameters: Samples for hyperparameter tuning
  :param Y_hyperparameters: Output variable for hyperparameter tuning
  :param output: The name of the output variable of interest
  :param imblearn_classes (deprecated): Classes dealing with imbalanced data.
  :param scorings: Scoring metrics
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :param force: whether the results are recalculated, although they exist
  :return None
  """
  
  print("\n---------------- Hyperparameters Eval ----------------\n")
  
  # https://www.econstor.eu/bitstream/10419/22569/1/tr56-04.pdf
  # https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
  # https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
  # https://scikit-learn.org/stable/modules/calibration.html#calibration
  #
  # SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
  # source: https://scikit-learn.org/dev/modules/svm.html#scores-probabilities
  # When the constructor option probability is set to True, class membership probability estimates
  #   (from the methods predict_proba and predict_log_proba) are enabled. In the binary case, the probabilities are 
  #  calibrated using Platt scaling 9: logistic regression on the SVM’s scores, fit by an additional cross-validation on the training data
  
  for imblearn_class in imblearn_classes:
    imblearn_class_descr = "None"
    if imblearn_class is not None:
      imblearn_class_descr = imblearn_class.__str__()
    
    for model_dict in models_list:
      model_name = model_dict["model_name"]
      model = model_dict["model"]
      param_grid = model_dict["param_grid"]
  
      if task_description == "ordinal_classification":
        model.set_imblearn_class(imblearn_class)
               
      for scoring in scorings:
        if not task_description.startswith("regression"):
          print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring+", imblearn="+imblearn_class_descr)
        else:
          print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring)
  
        output_filepath = os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+task_description, "hyperparameters-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+".csv")
        if not task_description.startswith("regression"):  
          output_filepath = os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+task_description, "hyperparameters-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".csv")
        
        if not os.path.exists(output_filepath) or force:
          counts_filepath = ""
          if not task_description.startswith("regression"):
            counts_filepath = os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+task_description, "hyperparameters-class_counts"+"_model="+model_name+"_output="+ output+"_imblearn="+imblearn_class_descr+".csv")
          perform_hyperparameters_eval_for_classification_or_regression(task_description, X_hyperparameters, Y_hyperparameters, model, param_grid, output_filepath, counts_filepath, imblearn_class, scoring, n_jobs)
 




def perform_prediction_for_classification_or_regression_with_generic_model(task_description, models_list, \
                                      X_train_test, Y_train_test, output, imblearn_classes, \
                                      scorings, n_jobs, force=False):
  """
  It is a wrapper function to perform the training and prediction parts in a classification or regression task.
  It iterates over:
  - different machine learning models
  - scoring metrics
  - classes dealing with imbalanced data

  :param task_description: task description, e.g. "binary_classification"
  :param models_list: a list of machine learning models
  :param X_train_test: Samples for train and test set
  :param Y_train_test: Output variable for train and test set
  :param output: The name of the output variable of interest
  :param imblearn_classes (deprecated): Classes dealing with imbalanced data.
  :param scorings: Scoring metrics
  :param n_jobs: nb threads to perform this task. Scikit-learn deals with it.
  :param force: whether the results are recalculated, although they exist
  :return None
  """
  

  print("\n---------------- Prediction Eval ----------------\n")
  
  for imblearn_class in imblearn_classes:
    imblearn_class_descr = "None"
    if imblearn_class is not None:
      imblearn_class_descr = imblearn_class.__str__()

    for model_dict in models_list:
      model_name = model_dict["model_name"]
      model = model_dict["model"]
      param_grid = model_dict["param_grid"]
      
      if task_description == "ordinal_classification":
        model.set_imblearn_class(imblearn_class)
      
      for scoring in scorings:
        if not task_description.startswith("regression"):
          print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring+", imblearn="+imblearn_class_descr)
        else:
          print("task="+task_description+", output="+output+", model="+model_name+", scoring="+scoring)

        output_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "prediction-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+".csv")
        if not task_description.startswith("regression"):  
          output_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "prediction-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".csv")
        if not os.path.exists(output_filepath) or force:
          hyperparameters_result_filepath = os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+task_description, "hyperparameters-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+".csv")
          counts_filepath = ""
          if not task_description.startswith("regression"):
            hyperparameters_result_filepath = os.path.join(consts.HYPERPARAMETERS_EVAL_RESULTS_FOLDER, "task="+task_description, "hyperparameters-results"+"_model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".csv")
            counts_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "prediction-class-counts"+"_model="+model_name+"_output="+ output+"_imblearn="+imblearn_class_descr+".csv")
          
          best_model_pickle_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "best-model="+model_name+"_output="+ output+"_scoring="+scoring+".pkl")
          if not task_description.startswith("regression"):
            best_model_pickle_filepath = os.path.join(consts.PREDICTION_EVAL_RESULTS_FOLDER, "task="+task_description, "best-model="+model_name+"_output="+ output+"_scoring="+scoring+"_imblearn="+imblearn_class_descr+".pkl")

          if not os.path.exists(best_model_pickle_filepath) or force:
            best_model = perform_classification_or_regression_prediction(task_description, X_train_test, Y_train_test, model, output_filepath, counts_filepath,  hyperparameters_result_filepath, imblearn_class, scoring, n_jobs)
            with open(best_model_pickle_filepath, 'wb') as f:
              pickle.dump(best_model, f)
          


