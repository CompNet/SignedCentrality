import math
import pandas
import numpy
import sys
import random
import csv
import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from util import get_matrix, scale_centrality
from igraph import *
from centrality.pyrwr.pyrwr.rwr import RWR
import consts

def perform_rwr(graph, graph_type):
    '''
    This method performs the SRWR algorithm to calculate the centrality values for
    the nodes of a given graph.

    :param graph: i-graph object
    :type graph: i-graph object
    :param graph_type: indicates the type of the graph (undirected or directed)
    :type graph_type: string
    '''

    rwr = RWR()

    f = open("RWR_temp.txt","w+")
    cursor = 0
    for i in graph.es:
        stringToWrite = str(i.source) + "\t" + str(i.target) + "\t" + str(graph.es['weight'][cursor]) + "\n"
        print(stringToWrite)
        f.write(stringToWrite)
        cursor += 1
    f.close()
    rwr.read_graph("RWR_temp.txt", graph_type)
    r = rwr.compute(seed, c, epsilon, max_iters)
    print("r = \n", r)
    rwr = []
    for value in r:
        rwr.append(r)
        
    graph.vs['rwr'] = rwr
    print(graph.vs['rwr'])
    return graph


#TESTS:

W = [[0    ,   0.8    ,  -0.1  ,  0.1 , 0.4 ],
     [1.0  ,   0    ,  -0.9  ,  0.3 , -0.1 ],
     [-0.9    ,  -0.6  ,   0    ,  -1 , -0.6 ],
     [0.9    ,  -0.6  ,   0.5    ,  0 , 0.2  ],
     [0.8  ,   0.3    ,   0.7  ,  0.3 , 0 ]]

edge_values = [(0, 1, 0.8), (0, 2, -0.1), (0, 3, 0.1), (0, 4, 0.4),
               (1, 0, 1.0), (1, 2, -0.9), (1, 3, 0.3), (1, 4, -0.1), 
               (2, 0 , -0.9), (2, 1, -0.6), (2, 3, -1), (2, 4, -0.6),
               (3, 0, 0.9), (3, 1, -0.6), (3, 2, 0.5), (3, 4, 0.2),
               (4, 0, 0.8), (4, 1, 0.3), (4, 2, 0.7), (4, 3, 0.3)]

edge = []
weights = []

for i in range(20):
    for j in range(2):
        edge.append(edge_values[i][j])
    weights.append(edge_values[i][2])

edges = [(i,j) for i,j in zip(edge[::2], edge[1::2])]

list1 = []
for i in range(len(edges)):
    list1.append((edges[i][0], edges[i][1]))

Gtest = Graph(directed=True)

Gtest.add_vertices(5)
Gtest.add_edges(list1)
Gtest.degree(mode="in")
Gtest.es['weight'] = weights

A = get_matrix(Gtest).toarray()
print(A)

perform_rwr(Gtest, "directed")
