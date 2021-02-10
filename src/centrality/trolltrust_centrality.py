'''
@author: alexandre
'''

import math
import pandas
import numpy
import sys
import random
import csv
#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
# Import train_test_split function
from sklearn.model_selection import train_test_split



import sys
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
        '''
        calculates a list of Reputation Values for each nodes
        '''


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
        graph.es['rep'] = rep
        
            
    def calculate_opt_values(graph, pi):
        '''
        calculates a list of Optimistim Values for each nodes
        '''

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
                
        graph.es['opt'] = opt


        

    def troll_trust(graph, beta, lambda1, iter_max, delta_min):
        '''
        returns a list pi containing centrality values for each nodes from the
        graph
        '''

        #W = graph.get_adjacency(type=GET_ADJACENCY_BOTH, eids=False)
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
        print("iter_number = ", iter_number)
        return piT2


    def logistic_regression(graph_10_percent, graph_90_percent, kernel):
        '''
        trains a regressor to predict the sign from the links of a graph
        '''

        df = pandas.DataFrame(graph_90_percent.es['opt'],columns=['opt'])
        df += pandas.DataFrame(graph_90_percent.es['rep'],columns=['rep'])
        Y = df.to_numpy()

        df2 = pandas.DataFrame(graph_10_percent.es['weight'],columns=['weight'])
        X = df2.to_numpy()

        scaler = StandardScaler()

        scaler.fit(X)
        X = scaler.transform(X)

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                            random_state=109)
        Y_train = Y_train.ravel()
        reg = svm.SVR(kernel=kernel)
        reg.fit(X_train, Y_train)

        Y_pred = reg.predict(X_test)

        print("R2 score:", metrics.r2_score(Y_test, Y_pred))
        return metrics.r2_score(Y_test, Y_pred)


    def choose_parameters(graph, iter_max, delta_min):
        '''
        returns the optimal parameters to perform the troll-trust algorithm on a
        given graph
        '''
        score = 0
        final_lambda1 = 0
        final_beta = 0
        for i in numpy.arange(0.01, 1, 0.01):
        
            lambda1 = i

            for j in numpy.arange(0.001, 1, 0.001):

                beta = j

                print("lambda1 = ", lambda1, "beta = ", beta)
                
                graph_10_percent = Graph(directed = True)
                graph_10_percent.add_vertices(len(graph.vs))
                graph_10_percent.degree(mode="in")

                
                graph_90_percent = graph
                
                edge_number_to_remove = len(graph.es) / 10

                W = get_matrix(graph_90_percent).toarray()
                print(W)
                edges_10_percent = []
                weights_10_percent = []
                edge_number_to_remove += 1
                print(edge_number_to_remove)
                
                while edge_number_to_remove > 0:
                    v1, v2 = random.sample(range(graph.vcount()), 2)
                    if (graph_90_percent.get_eid(v1, v2, directed=True, error=False) != -1):
                        print(graph.get_eid(v1, v2, directed=True, error=False))
                        edges_10_percent.append((v1, v2))
                        weights_10_percent.append(W[v1][v2])                                   
                        graph_90_percent.delete_edges([(v1, v2)])
                        edge_number_to_remove -= 1
                print(edges_10_percent)

                graph_10_percent.add_edges(edges_10_percent)
                graph_10_percent.es['weight'] = weights_10_percent
                print("ok")
                pi = TrollTrust.troll_trust(graph_90_percent, beta, lambda1, iter_max, delta_min)

                print("pi:")
                
                for c1 in range(len(graph.vs)):
                    print(c1, " => ", pi[c1])
                                
                TrollTrust.calculate_rep_values(graph_90_percent, pi)
                TrollTrust.calculate_opt_values(graph_90_percent, pi)
                kernel = consts.PREDICTION_KERNEL_LINEAR
                new_score = TrollTrust.logistic_regression(graph_10_percent, graph_90_percent, kernel)
                if score < new_score:
                    score = new_score
                    final_lambda1 = lambda1
                    final_beta = beta

        return final_lambda1, final_beta

# MAIN:

''' test values : '''
W = [[0    ,   0    ,  -0.1  ,  0.1],
     [1.0  ,   0    ,  -0.9  ,  0  ],
     [0    ,  -0.6  ,   0    ,  0  ],
     [0.8  ,   0    ,   0.7  ,  0  ]]

edge_values = [(0, 2, -0.1), (0, 3, 0.1), (1, 0, 1.0), (1, 2, -0.9), (2, 1, -0.6),
              (3, 0, 0.8), (3, 2, 0.7)]

edge = []
weights = []

for i in range(7):
    for j in range(2):
        edge.append(edge_values[i][j])
    weights.append(edge_values[i][2])

edges = [(i,j) for i,j in zip(edge[::2], edge[1::2])]

list1 = []
for i in range(len(edges)):
    list1.append((edges[i][0], edges[i][1]))

Gtest = Graph(directed=True)

Gtest.add_vertices(4)
Gtest.add_edges(list1)
Gtest.degree(mode="in")
Gtest.es['weight'] = weights

A = get_matrix(Gtest).toarray()
print(A)


iter_max = 1000
delta_min = 0

TrollTrust.choose_parameters(Gtest, iter_max, delta_min)


