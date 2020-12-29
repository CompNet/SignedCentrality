'''

@author: alexandre
'''

import math
import numpy
import sys
import random
import csv
#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from prediction.regression import perform_regression
from __init__ import CentralityMeasure
import consts


class TrollTrust(CentralityMeasure):
    

    def generate_feature_and_output(graph):
        '''
        generates two .csv file based on a graph: one for the features and one for the ouput
        '''

        with open('feature.csv', 'w', newline='') as csvfile:
            fieldnames = [consts.STATS_NB_NODES, consts.STATS_POS_NEG_RATIO,
                          consts.STATS_POS_PROP, consts.STATS_NEG_PROP,
                          consts.STATS_SIGNED_TRIANGLES, consts.STATS_LARGEST_EIGENVALUE]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({consts.STATS_NB_NODES : graph.size, consts.STATS_POS_NEG_RATIO : ,
                             consts.STATS_POS_PROP : , consts.STATS_NEG_PROP : ,
                             consts.STATS_SIGNED_TRIANGLES : , consts.STATS_LARGEST_EIGENVALUE})


    def truncate_graph(graph):
        '''
        returns a graph with only 90% of the links from the entry graph and another
        graph with only the 10% left
        '''
        truncatedWeightMatrix = graph.W
        remainingWeightMatrix = [[0] * graph.size] * graph.size
        
        remainingGraphLinksNumber = linksToRemoveNumber = (graph.links_number * 10) // 100

        truncateGraphLinksNumber == graph.links_number - linksToRemoveNumber
        
        while linksToRemoveNumber < 0:
            cursorI = random.randint(0, graph.size -1)
            cursorJ = random.randint(0, graph.size -1)
            if truncatedWeightMatrix[cursorI][cursorJ] != 0:
                remainingWeightMatrix[cursorI][cursorJ] = truncatedWeightMatrix[cursorI][cursorJ]
                truncatedWeightMatrix[cursorI][cursorJ] = 0
                linksToRemoveNumber -= 1

        truncatedGraph = graph(graph.size, truncatedWeightMatrix, truncateGraphLinksNumber)
        remainingGraph = graph(graph.size, remainingWeightMatrix, remainingGraphLinksNumber)


        return truncatedGraph, remainingGraph


    def calculateRepValues(graph, pi):
        '''
        returns a list of Reputation Values for each nodes
        '''

        rep = [0 for w in range(graph.size)]

        for i in range(graph.size):

            o1, o2 = 0, 0, 0, 0

            for j in range(graph.size):

                if graph.W[i][j] > 0:
                    o1 += pi[j]
                elif graph.W[i][j] < 0:
                    o2 += pi[j]

            rep[i] = (o1 - o2) / (o1 + o2)

            
    def calculateOptValues(graph, pi):
        '''
        returns a list of Optimistim Values for each nodes
        '''

        opt = [0 for w in range(graph.size)]
        
        for i in range(graph.size):

            o1, o2 = 0, 0, 0, 0

            for j in range(graph.size):

                if graph.W[j][i] > 0:
                    o1 += pi[j]
                elif graph.W[j][i] < 0:
                    o2 += pi[j]

            opt[i] = (o1 - o2) / (o1 + o2)
                

    def chooseParameters(graph, iter_max, delta_min):        
        
        for i in numpy.arange(0.01, 1, 0.01):
        
            lambda1 = i

            for j in numpy.arange(0.001, 1, 0.001):

                beta = j

                pi = troll_trust(graph, beta, lambda1, iter_max, delta_min)
                print("lambda1 = ", lambda1, "beta = ", beta)
                print("pi:")
                
                for c1 in range(Gtest.size):
                    print(c1, " => ", pi[c1])
                if((0.6 <= pi[0] < 0.7) and (0.5 <= pi[1] < 0.6) and (0.4 <= pi[2] < 0.5) and (0.5 <= pi[3] < 0.6)):
                    sys.exit("Approximation found!")
                print("\n")


    def troll_trust(graph, beta, lambda1, iter_max, delta_min):
        '''
        returns a list pi containing centrality values for each nodes from the graph
        '''

        W = graph.get_adjacency(type=GET_ADJACENCY_BOTH, eids=False)
        
        iter_number = 0

        lambda0 = -(math.log(beta/(1-beta)))
        
        piT1 = [beta for w in range(graph.vs.size)]
        piT2 = [beta for w in range(graph.vs.size)]

        delta = math.inf

        while (abs(delta) > delta_min) and (iter_number < iter_max):

            for i in range(graph.vs.size):
                
                n1, n2, d1, d2 = 0, 1, 0, 1
                
                for j in range(graph.vs.size):
                    if W[j][i] != 0:
                        n1 += piT1[j] * (1 / (1 + math.e **(lambda0 - lambda1 * W[i][j])))
                        n2 *= 1-piT1[j]
                        d1 += piT1[j]
                        d2 *= 1-piT1[j]
                piT2[i] = (n1 + beta * n2) / (d1 + d2)

            delta = 0
            
            for k in range(graph.vs.size):
                delta += piT2[k] - piT1[k]
            delta /= graph.vs.size
            
            for l in range(graph.vs.size):
                piT1[l] = piT2[l]
            iter_number += 1
        print("iter_number = ", iter_number)
        return piT2


    def logistic_regression(graph, features, output, kernel):
        '''
        trains a regressor to predict the sign from the links of a graph
        '''
        reg = svm.SVR(kernel=kernel)
        reg.fit(X_train, Y_train)
        kernel = consts.PREDICTION_KERNEL_LINEAR

        # =======================================================
        # Test: Predict the response for test dataset
        # =======================================================
        Y_pred = reg.predict(X_test)

        # =======================================================
        # Metrics
        # =======================================================
        print("R2 score:", metrics.r2_score(Y_test, Y_pred))




# MAIN:


print("#############################################################")
print("#############################################################")
print("#########                                           #########")
print("#########                 Troll-trust               #########")
print("#########                                           #########")
print("#############################################################")
print("#############################################################")


''' test values : '''
W = [[0    ,   0    ,  -0.1  ,  0.1],
     [1.0  ,   0    ,  -0.9  ,  0  ],
     [0    ,  -0.6  ,   0    ,  0  ],
     [0.8  ,   0    ,   0.7  ,  0  ]]

Gtest

iter_max = 1000
delta_min = 0

chooseParameters(Gtest, iter_max, delta_min)


