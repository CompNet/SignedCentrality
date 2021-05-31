'''
@author: alexandre
'''

import math
import pandas
import numpy as np
import sys
import random
import csv
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from igraph import *
from descriptors import GraphDescriptor
from descriptors.centrality import CentralityMeasure
import consts
from bct.algorithms.centrality import diversity_coef_sign
from bct.algorithms import modularity_finetune_dir
from util import get_matrix

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class diversity_coef_centrality(GraphDescriptor):
    """
    This class is used to compute Weight-conserving Characterization of Complex
    Functional Brain Networks centralities.
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        Compute the Weight-conserving Characterization of Complex
        Functional Brain Networks centrality
        """
        return diversity_coef_centrality.undirected(graph, **kwargs)

    @staticmethod
    def undirected(graph, **kwargs):
        """
        Compute the Weight-conserving Characterization of Complex
        Functional Brain Networks centrality
        """
        W = get_matrix(graph).toarray()
        ci, Q = modularity_finetune_dir(W)
        Hpos, Hneg = diversity_coef_sign(W, ci)
        return Hpos

