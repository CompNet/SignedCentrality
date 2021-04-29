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

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from igraph import *

from descriptors import GraphDescriptor

from centrality import CentralityMeasure
import consts

from bct.algorithms.centrality import diversity_coef_sign

from bct.algorithms import modularity_finetune_dir

from util import get_matrix


class diversity_coef_centrality(GraphDescriptor):

    @staticmethod
    def perform_diversity_coef_centrality(graph):
        W = get_matrix(graph).toarray()
        ci, Q = modularity_finetune_dir(W)
        Hpos, Hneg = diversity_coef_sign(W, ci)
        print(Hpos)
        print(Hneg)



#Main for tests:

##W = [[0    ,   0.8    ,  -0.1  ,  0.1 , 0.4 ],
##     [1.0  ,   0    ,  -0.9  ,  0.3 , -0.1 ],
##     [-0.9    ,  -0.6  ,   0    ,  -1 , -0.6 ],
##     [0.9    ,  -0.6  ,   0.5    ,  0 , 0.2  ],
##     [0.8  ,   0.3    ,   0.7  ,  0.3 , 0 ]]
##
##edge_values = [(0, 1, 0.8), (0, 2, -0.1), (0, 3, 0.1), (0, 4, 0.4),
##               (1, 0, 1.0), (1, 2, -0.9), (1, 3, 0.3), (1, 4, -0.1), 
##               (2, 0 , -0.9), (2, 1, -0.6), (2, 3, -1), (2, 4, -0.6),
##               (3, 0, 0.9), (3, 1, -0.6), (3, 2, 0.5), (3, 4, 0.2),
##               (4, 0, 0.8), (4, 1, 0.3), (4, 2, 0.7), (4, 3, 0.3)]
##
##edge = []
##weights = []
##
##for i in range(20):
##    for j in range(2):
##        edge.append(edge_values[i][j])
##    weights.append(edge_values[i][2])
##
##edges = [(i,j) for i,j in zip(edge[::2], edge[1::2])]
##
##list1 = []
##for i in range(len(edges)):
##    list1.append((edges[i][0], edges[i][1]))
##
##Gtest = Graph(directed=True)
##
##Gtest.add_vertices(5)
##Gtest.add_edges(list1)
##Gtest.degree(mode="in")
##Gtest.es['weight'] = weights
##
##
##
##diversity_coef_centrality.perform_diversity_coef_centrality(Gtest)
