'''
Created on Feb 1, 2022

@author: nejat
'''


import os
import consts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split





def plot_regression_true_pred_error_values(X_train_test, Y_train_test, model, model_name, output):
  """
  It plots the regression results for a better understanding of the performances of the regression models.
  - true values vs squared error
  - predicted values vs squared error
  - true values vs predicted values
  
  :param X_train_test: Samples for train and test set
  :param Y_train_test: Output variable for train and test set
  :param model: a trained regression model
  :param model_name: the name of the regression model
  :param output: The name of the output variable of interest
  :return None
  """
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test, test_size=0.3, random_state=0)
  #model = reg.fit(X_train, Y_train)
  #score = model.score(X_test, Y_test) # r2
  #print(score)
  nb_test_samples = Y_test.shape[0]
  print(nb_test_samples)
  print("----")
  print(Y_test[5])
  print(model.predict(X_test[5,].reshape(1, -1)))
  errors = []
  pred_values = []
  true_values = []
  for i in range(nb_test_samples):
    pred_value = model.predict(X_test[i,].reshape(1, -1))[0]
    pred_values.append(pred_value)
    true_value = Y_test[i]
    true_values.append(true_value)
    error = (pred_value-true_value)*(pred_value-true_value)
    errors.append(error)
  #print("!!!")
  #print(pred_values[1:40])
  #plt.ticklabel_format(style='plain')    # to prevent scientific notation.
  plt.plot(true_values,errors,"ob") # ob = type de points "o" ronds, "b" bleus
  plt.xlabel('True values')
  plt.ylabel('Squared error')
  plt.title(model_name +" regression for "+output)
  plt.savefig(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression", "model="+model_name+"_true_vs_errors_for="+output+".pdf"), format='pdf')
  #
  plt.clf()
  #plt.ticklabel_format(style='plain')    # to prevent scientific notation.
  plt.plot(pred_values,errors,"ob") # ob = type de points "o" ronds, "b" bleus
  plt.xlabel('Predicted values')
  plt.ylabel('Squared error')
  plt.title(model_name +" regression for "+output)
  plt.savefig(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression", "model="+model_name+"_pred_vs_errors_for="+output+".pdf"), format='pdf')
  #
  plt.clf()
  #plt.ticklabel_format(style='plain')    # to prevent scientific notation.
  plt.plot(true_values,pred_values,"ob") # ob = type de points "o" ronds, "b" bleus
  plt.xlabel('True values')
  plt.ylabel('Predicted values')
  plt.title(model_name +" regression for "+output)
  plt.savefig(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression", "model="+model_name+"_true_vs_pred_for="+output+".pdf"), format='pdf')

  df_true_values = pd.DataFrame({output:true_values})
  df_true_values.to_csv(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression", "model="+model_name+"_true_values_for="+output+".csv"), sep=";", index=False)
  df_pred_values = pd.DataFrame({output:pred_values})
  df_pred_values.to_csv(os.path.join(consts.PREDICTION_PLOTS_FOLDER, "task="+"regression", "model="+model_name+"_pred_values_for="+output+".csv"), sep=";", index=False)

 