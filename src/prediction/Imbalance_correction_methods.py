'''
Created on Apr 6, 2021

@author: alexandre
'''

import itertools

import consts
import centrality.runner
import stats.runner
import collect.collect_features
import collect.collect_outputs
import prediction.classification
import prediction.regression

import prediction.random_forest_classification

import prediction.feature_ablation
from prediction.hyper_parameters import compare_hyper_parameters


from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

import numpy as np
import matplotlib.pyplot as plt


imbalance_methods = [RandomUnderSampler(), NearMiss(version=1), NearMiss(version=3),
                        TomekLinks(), EditedNearestNeighbours(), OneSidedSelection(),
                        NeighbourhoodCleaningRule(), RandomOverSampler(), SMOTE(),
                        BorderlineSMOTE(), SVMSMOTE(), ADASYN()]

def test_best_imbalance_method(classifier, features, output, iterations):

    results = []
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    best_imbalance_method = False
    best_f1_score = 0

    for imbalanceMethod in imbalance_methods:
        
        f1_score = 0
        accuracy = 0
        precision = 0
        recall = 0
        
        for i in iterations:
            metrics = prediction.classifier.perform_classification(features, output, kernel,
                                                                       imbalance_correction_method=imbalanceMethod)
            f1_score += metrics[1]['f1_score']
            accuracy += metrics[1]['accuracy_score']
            precision += metrics[1]['precision_score']
            recall += metrics[1]['recall_score']

        f1_score /= iterations
        accuracy /= iterations
        precision /= iterations
        recall /= iterations
        perfs = [type(imbMethod), f1_score, accuracy, precision, recall]
        results.append(perfs)

        if best_f1_score < f1_score:
            best_f1_score = f1_score
            best_imbalance_method = imbalanceMethod

    for i in results:
        print(i)

    n_groups = len(imbalance_methods)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 1 / len(imbalance_methods)
    opacity = 1

    rects1 = plt.bar(index, f1_scores, bar_width,
    alpha=opacity,
    color='b',
    label='f1')

    rects2 = plt.bar(index + bar_width, accuracy_scores, bar_width,
    alpha=opacity,
    color='y',
    label='accuracy')

    rects3 = plt.bar(index + bar_width, precision_scores, bar_width,
    alpha=opacity,
    color='r',
    label='precision')

    rects4 = plt.bar(index + bar_width, recall_scores, bar_width,
    alpha=opacity,
    color='g',
    label='recall')

    plt.ylabel('Scores')
    plt.title('Imbalance correction method influance on '+ classifier +' scores')
    plt.xticks(index + bar_width, (imbalance_methods))
    plt.legend()

    plt.tight_layout()
    plt.show()

    return results
