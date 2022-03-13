'''
@author: alexandre
'''

import pandas as pd
import numpy as np
import sys
import os
from descriptors import GraphDescriptor

# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from util import get_matrix
from igraph import *
import consts


class TrollTrust(GraphDescriptor):
    """
    Contain methods computing measures of Troll-Trust centrality on signed graphs
    """

    @staticmethod
    def calculate_rep_values(graph, pi):
        """This method calculates a list of Reputation values for each nodes.

        :param graph: i-graph object
        :type graph: i-graph object
        :param pi: list containing Troll-Trust centrality values for each nodes
        :type pi: int list
        """

        rep = [0 for w in range(len(graph.vs))]
        #W = get_matrix(graph).toarray()

        for vertex in range(len(graph.vs)):
            in_neighbors = graph.neighbors(vertex, mode="in")
                    
            o1, o2 = 0, 0
            for neigh in in_neighbors:
                weight = graph.es[graph.get_eid(vertex, neigh, error=False)]['weight']
                if weight > 0:
                    o1 += pi[neigh]
                elif weight < 0:
                    o2 += pi[neigh]
                
            if (o1+o2 == 0):
                rep[vertex] = 0
            else:                
                rep[vertex] = (o1 - o2) / (o1 + o2)
        return(rep)
      
        
    @staticmethod
    def calculate_opt_values(graph, pi):
        """This method calculates a list of Optimization values for each nodes.

        :param graph: i-graph object
        :type graph: i-graph object
        :param pi: list containing Troll-Trust centrality values for each nodes
        :type pi: int list
        """

        opt = [0 for w in range(len(graph.vs))]
        W = get_matrix(graph).toarray()
        
        for vertex in range(len(graph.vs)):
            out_neighbors = graph.neighbors(vertex, mode="out")
            
            o1, o2 = 0, 0
            for neigh in out_neighbors:
                if W[vertex][neigh] > 0:
                    o1 += pi[neigh]
                elif W[vertex][neigh] < 0:
                    o2 += pi[neigh]
            if (o1+o2 == 0):
                opt[vertex] = 0
            else:                
                opt[vertex] = (o1 - o2) / (o1 + o2)
                
        return(opt)
        

    @staticmethod
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
                in_neighbors = graph.neighbors(i, mode="in")
                
                for j in in_neighbors:
                    #if (graph.get_eid(i, j, directed=True, error=False) != -1):
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
      
      
      

    @staticmethod
    def read_beta_lambda1_parameters():
        '''
        This method returns the optimal parameters to perform the troll-trust algorithm on a
        given graph

        :param graph: i-graph object
        :type graph: i-graph object
        :param iter_max: indicates the maximum number of iteration of the while loop
        :type iter_max: int
        :param delta_min: indicates the acceptable convergence limit delta 
        :type delta_min: float
        :param lambda1_step: indicates at which step the algorithm will go through the values for lambda1
        :type lambda1_step: float
        :param beta_step: indicates at which step the algorithm will go through the values for beta
        :type beta_step: float
        '''
        
        filepath = os.path.join(consts.OUT_FOLDER,"best_trolltrust_parameters.csv")
        df = pd.read_csv(filepath, usecols=['lambda1', 'beta'], sep=";")
        return df["lambda1"].to_numpy()[0], df["beta"].to_numpy()[0]
      
      

    @staticmethod
    def perform_troll_trust(graph, iter_max = 10000, delta_min = 0, lambda1_step = 0.01, beta_step = 0.01):
        '''
        This method returns the best centrality values for the nodes of a graph

        :param graph: i-graph object
        :type graph: i-graph object
        :param iter_max: indicates the maximum number of iteration of the while loop
        :type iter_max: int
        :param delta_min: indicates the acceptable convergence limit delta 
        :type delta_min: float
        :param lambda1_step: indicates at which step the algorithm will go through the values for lambda1
        :type lambda1_step: float
        :param beta_step: indicates at which step the algorithm will go through the values for beta
        :type beta_step: float
        '''
        
        lambda_1, beta = TrollTrust.read_beta_lambda1_parameters()
        pi = TrollTrust.troll_trust(graph, beta, lambda_1, iter_max, delta_min)
        graph.vs['pi'] = pi
        
        print("!!!!!!!!!!!! over !!!!!!!!!!!!")
        print("beta =", beta, "; lambda1 =", lambda_1)
        print("pi = ", graph.vs['pi'])
    
        return pi

    @staticmethod
    def perform(graph, **kwargs):
        """
        Compute the Troll Trust centrality.
        """
        return TrollTrust.perform_troll_trust(graph, **kwargs)

    @staticmethod
    def undirected(graph, **kwargs):
        """
        Compute the Troll Trust centrality.
        """

        return TrollTrust.perform(graph, **kwargs)




# TESTS:

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



##W = [[0    ,   0.8    ,  -0.1  ,  0.1 , 0.4 ],
##     [1.0  ,   0    ,  -0.9  ,  0.3 , -0.1 ],
##     [-0.9    ,  -0.6  ,   0    ,  -1 , -0.6 ],
##     [0.9    ,  -0.6  ,   0.5    ,  0 , 0.2  ],
##     [0.8  ,   0.3    ,   0.7  ,  0.3 , 0 ]]
##
##edge_values = [(0, 1, 0.8), (0, 2, -0.1), (0, 3, 0.1), (0, 4, 0.4), (0, 5, 0.4), (0, 6, 0.4), (0, 7, 0.4), (0, 8, 0.4), (0, 9, 0.4),
##               (1, 0, 1.0), (1, 2, -0.9), (1, 3, 0.3), (1, 4, -0.1), (1, 5, -0.1), (1, 6, -0.1), (1, 7, -0.1), (1, 8, -0.1), (1, 9, -0.1),
##               (2, 0 , -0.9), (2, 1, -0.6), (2, 3, -1), (2, 4, -0.6), (2, 5, -0.6), (2, 6, -0.6), (2, 7, -0.6), (2, 8, -0.6), (2, 9, -0.6),
##               (3, 0, 0.9), (3, 1, -0.6), (3, 2, 0.5), (3, 4, 0.2), (3, 5, 0.2), (3, 6, 0.2), (3, 7, 0.2), (3, 8, 0.2), (3, 9, 0.2),
##               (4, 0, 0.8), (4, 1, 0.3), (4, 2, 0.7), (4, 3, 0.3), (4, 5, 0.2), (4, 6, 0.2), (4, 7, 0.2), (4, 8, 0.2), (4, 9, 0.2),
##               (5, 0, 0.8), (5, 1, 0.3), (5, 2, 0.7), (5, 3, 0.3), (5, 4, 0.2), (5, 6, 0.2), (5, 7, 0.2), (5, 8, 0.2), (5, 9, 0.2),
##               (6, 0, 0.8), (6, 1, 0.3), (6, 2, 0.7), (6, 3, 0.3), (6, 4, 0.2), (6, 5, 0.2), (6, 7, 0.2), (6, 8, 0.2), (6, 9, 0.2),
##               (7, 0, 0.8), (7, 1, 0.3), (7, 2, 0.7), (7, 3, 0.3), (7, 5, 0.2), (7, 6, 0.2), (7, 8, 0.2), (7, 9, 0.2),
##               (8, 0, 0.8), (8, 1, 0.3), (8, 2, 0.7), (8, 3, 0.3), (8, 4, 0.2), (8, 5, 0.2), (8, 6, 0.2), (8, 7, 0.2), (8, 9, 0.2),
##               (9, 0, 0.8), (9, 1, 0.3), (9, 2, 0.7), (9, 3, 0.3), (9, 5, 0.2), (9, 6, 0.2), (9, 7, 0.2), (9, 8, 0.2), (9, 4, 0.2)]
##
##edge = []
##weights = []
##
##for i in range(89):
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
##Gtest.add_vertices(10)
##Gtest.add_edges(list1)
##Gtest.degree(mode="in")
##Gtest.es[consts.EDGE_WEIGHT_ATR] = weights
##
##A = get_matrix(Gtest).toarray()
##print(A)
##
##
##iter_max = 1000
##delta_min = 0
##
##TrollTrust.perform_troll_trust(Gtest, iter_max, delta_min, 0.1, 0.1)
##
