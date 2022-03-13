'''
Created on Feb 1, 2022

@author: nejat
'''



import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score



# ========================================================================
# TODO: Add logistic regression model from  statsmodels by creating sklearn wrapper class
# source: https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
# ========================================================================




class OrdinalClassifier():
  """
  We build here a generic ordinal classifier,
   which accepts any binary classifier, based on this paper:
  https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf.
  
  Note that we can still use this custom classifier in a scikit-learn pipeline.
  """
  
  def __init__(self, clf, imblearn_class=None):
    self.clf = clf
    self.clfs = {}
    if imblearn_class is not None:
      self.imblearn_class = imblearn_class
      
  def set_imblearn_class(self, imblearn_class):
    self.imblearn_class = imblearn_class
    
    
  # def get_params(self, deep):
  #   return(self.clf.get_params(deep))
  
  # def clone(self):
  #   return(clone(self.clf))
        
  def set_params(self, **params):
    self.clf.set_params(**params)
    
    
  def fit(self, X, y):
    self.unique_class = np.sort(np.unique(y))
    if self.unique_class.shape[0] > 2:
      for i in range(self.unique_class.shape[0] - 1):
        # for each k - 1 ordinal value we fit a binary classification problem
        binary_y = (y > self.unique_class[i]).astype(np.uint8)
        clf = clone(self.clf)
        #X2, Y2 = self.imblearn_class.fit_resample(X, binary_y)
        #clf.fit(X2, Y2)
        clf.fit(X, binary_y)
        self.clfs[i] = clf


  def predict_proba(self, X):
    # to get predicted probability of each class: first we get all prediction probability from
    #   all of our classifiers that stored on clfs 
    #   after that simply enumerate all possible class label and append its prediction 
    #   to our predicted list, after that, return it as a numpy array
    clfs_predict = {k: v.predict_proba(X) for k, v in self.clfs.items()}
    #print(clfs_predict)
    predicted = []
    for i, y in enumerate(self.unique_class):
      if i == 0:
        # V1 = 1 - Pr(y > V1)
        predicted.append(1 - clfs_predict[i][:, 1])
      elif y in clfs_predict:
        # Vi = Pr(y > Vi-1) - Pr(y > Vi)
        predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
      else:
        # Vk = Pr(y > Vk-1)
        predicted.append(clfs_predict[i - 1][:, 1])
    # >>> in_arr1 = np.array([ 1, 2, 3] )
    # >>> in_arr2 = np.array([ 4, 6, 5] )
    # >>> out_arr = np.vstack((in_arr1, in_arr2))
    # array([[1, 2, 3],
    #        [4, 6, 5]])
    # >>> out_arr = np.vstack((in_arr1, in_arr2))
    #  array([[1, 4],
       # [2, 6],
       # [3, 5]])
    return np.vstack(predicted).T


  def predict(self, X):
    # >>> out_arr
    #  array([[1, 4],
       # [2, 6],
       # [3, 5]])
    # >>> np.argmax(out_arr, axis=1)
    # array([1, 1, 1])
    return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]
      

  def score(self, X, y):
    pred_y_vals = self.predict(X)
    return accuracy_score(y, pred_y_vals, sample_weight=None)
  
