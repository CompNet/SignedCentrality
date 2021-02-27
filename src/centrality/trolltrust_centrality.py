'''
@author: alexandre
'''

import math
import pandas
import numpy
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

from util import get_matrix, scale_centrality
from igraph import *

from prediction.regression import perform_regression
from __init__ import CentralityMeasure
import consts


class TrollTrust(CentralityMeasure):

    def calculate_rep_values(graph, pi):
        """This method calculates a list of Reputation values for each nodes.

        :param graph: i-graph object
        :type graph: i-graph object
        :param pi: list containing Troll-Trust centrality values for each nodes
        :type pi: int list
        """

        rep = [0 for w in range(len(graph.vs))]

        W = get_matrix(graph).toarray()

        for i in range(len(graph.vs)):
        
            o1, o2 = 0, 0

            for j in range(len(graph.vs)):

                if W[i][j] > 0:
                    o1 += pi[j]
                elif W[i][j] < 0:
                    o2 += pi[j]
                
            if (o1+o2 == 0):
                rep[i] = 0
            else:                
                rep[i] = (o1 - o2) / (o1 + o2)
        graph.vs['rep'] = rep
        
            
    def calculate_opt_values(graph, pi):
        """This method calculates a list of Optimization values for each nodes.

        :param graph: i-graph object
        :type graph: i-graph object
        :param pi: list containing Troll-Trust centrality values for each nodes
        :type pi: int list
        """

        opt = [0 for w in range(len(graph.vs))]

        W = get_matrix(graph).toarray()
        
        for i in range(len(graph.vs)):

            o1, o2 = 0, 0

            for j in range(len(graph.vs)):

                if W[j][i] > 0:
                    o1 += pi[j]
                elif W[j][i] < 0:
                    o2 += pi[j]
            if (o1+o2 == 0):
                opt[i] = 0
            else:                
                opt[i] = (o1 - o2) / (o1 + o2)
                
        graph.vs['opt'] = opt
        

    def troll_trust(graph, beta, lambda1, iter_max, delta_min):
        """This method returns a list pi containing centrality values for each
        nodes from the graph.

        :param graph: i-graph object
        :type graph: i-graph object
        :param beta: 
        :type beta: float
        :param lambda1:
        :type lambda1: float
        :param iter_max: indicates the maximum number of iteration of the while loop
        :type iter_max: int
        :param delta_min: indicates the acceptable convergence limit delta 
        :type delta_min: float
        """

        W = get_matrix(graph).toarray()
        iter_number = 0

        lambda0 = -(math.log(beta/(1-beta)))
        
        piT1 = [beta for w in range(len(graph.vs))]
        piT2 = [beta for w in range(len(graph.vs))]

        delta = math.inf

        while (abs(delta) > delta_min) and (iter_number < iter_max):

            for i in range(len(graph.vs)):
                
                n1, n2, d1, d2 = 0, 1, 0, 1
                
                for j in range(len(graph.vs)):
                    if (graph.get_eid(i, j, directed=True, error=False) != -1):
                        n1 += piT1[j] * (1 / (1 + math.e **(lambda0 - lambda1 * W[j][i])))
                        n2 *= 1-piT1[j]
                        d1 += piT1[j]
                        d2 *= 1-piT1[j]
                piT2[i] = (n1 + beta * n2) / (d1 + d2)

            delta = 0
            
            for k in range(len(graph.vs)):
                delta += piT2[k] - piT1[k]
            delta /= len(graph.vs)
            
            for l in range(len(graph.vs)):
                piT1[l] = piT2[l]
            iter_number += 1
            
        return piT2


    def logistic_regression(graph, kernel):
        '''
        This method trains a regressor to predict the sign from the links of a
        graph

        :param graph: i-graph object
        :type graph: i-graph object
        :param kernel: indicates the kernel type wanted to perform the regression
        :type kernel: string
        '''

        opt_source = []
        rep_source = []
        opt_target = []
        rep_target = []

        for j in graph.es:
            opt_source.append(graph.vs['opt'][j.source])
            rep_source.append(graph.vs['rep'][j.source])
            opt_target.append(graph.vs['opt'][j.target])
            rep_target.append(graph.vs['rep'][j.target])

        df_x = pandas.DataFrame({
            'opt_source' : opt_source,
            'rep_source' : rep_source,
            'opt_target' : opt_target,
            'rep_target' : rep_target
        })
        
        X = df_x.to_numpy()

        df_y = pandas.DataFrame({
            'weights' : graph.es['weight']
        })

        Y = df_y.to_numpy()

        scaler = StandardScaler()

        scaler.fit(X)

        Y = Y.ravel()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, train_size=0.9, random_state=109, shuffle=True)

        reg = svm.SVR(kernel=kernel)
        reg.fit(X_train, Y_train)

        Y_pred = reg.predict(X_test)

        print("R2 score:", metrics.r2_score(Y_test, Y_pred))
        return metrics.r2_score(Y_test, Y_pred)


    def choose_parameters(graph, iter_max, delta_min):
        '''
        This method returns the optimal parameters to perform the troll-trust algorithm on a
        given graph

        :param graph: i-graph object
        :type graph: i-graph object
        :param iter_max: indicates the maximum number of iteration of the while loop
        :type iter_max: int
        :param delta_min: indicates the acceptable convergence limit delta 
        :type delta_min: float
        '''
        score = 0
        final_lambda1 = 0
        final_beta = 0
        for i in numpy.arange(0.01, 1, 0.01):
        
            lambda1 = i

            for j in numpy.arange(0.001, 1, 0.001):

                beta = j

                print("lambda1 = ", lambda1, "beta = ", beta)
                
                pi = TrollTrust.troll_trust(graph, beta, lambda1, iter_max, delta_min)

                print("pi:")
                
                for c1 in range(len(graph.vs)):
                    print(c1, " => ", pi[c1])
                                

                TrollTrust.calculate_rep_values(graph, pi)
                TrollTrust.calculate_opt_values(graph, pi)

                kernel = consts.PREDICTION_KERNEL_LINEAR

                new_score = TrollTrust.logistic_regression(graph, kernel)

                if score < new_score:
                    score = new_score
                    final_lambda1 = lambda1
                    final_beta = beta

        return final_lambda1, final_beta


    def perform_troll_trust(graph, iter_max, delta_min):
        '''
        This method returns the best centrality values for the nodes of a graph

        :param graph: i-graph object
        :type graph: i-graph object
        :param iter_max: indicates the maximum number of iteration of the while loop
        :type iter_max: int
        :param delta_min: indicates the acceptable convergence limit delta 
        :type delta_min: float
        '''
        
        lambda_1, beta = TrollTrust.choose_parameters(graph, iter_max, delta_min)
        pi = TrollTrust.troll_trust(graph, beta, lambda1, iter_max, delta_min)
        graph.vs['pi'] = pi
        
        return graph

# MAIN:

##''' test values : '''
##W = [[0    ,   0    ,  -0.1  ,  0.1],
##     [1.0  ,   0    ,  -0.9  ,  0  ],
##     [0    ,  -0.6  ,   0    ,  0  ],
##     [0.8  ,   0    ,   0.7  ,  0  ]]
##
##edge_values = [(0, 2, -0.1), (0, 3, 0.1), (1, 0, 1.0), (1, 2, -0.9), (2, 1, -0.6),
##              (3, 0, 0.8), (3, 2, 0.7)]
##
##edge = []
##weights = []
##
##for i in range(7):
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
##Gtest.add_vertices(4)
##Gtest.add_edges(list1)
##Gtest.degree(mode="in")
##Gtest.es['weight'] = weights
##
##A = get_matrix(Gtest).toarray()
##print(A)
##
##
##iter_max = 1000
##delta_min = 0
##
##TrollTrust.perform_troll_trust(Gtest, iter_max, delta_min)
##

