
import math
import numpy
import sys
import random

class graph:   
    
    def __init__(self, size, W, links_number):
        self.size = size
        self.links_number = links_number
        self.W = W

    def display(self):
        print("\n matrix values:")
        for i in range(self.size):
            for j in range(self.size):
                print(self.W[i][j], "  ")
            print('\n')

    def setMatrix(W):
        self.W = W



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

            pi = troll_trust(Gtest, beta, lambda1, iter_max, delta_min)
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

    iter_number = 0

    lambda0 = -(math.log(beta/(1-beta)))
    
    piT1 = [beta for w in range(graph.size)]
    piT2 = [beta for w in range(graph.size)]

    delta = math.inf

    while (abs(delta) > delta_min) and (iter_number < iter_max):

        for i in range(graph.size):
            
            n1, n2, d1, d2 = 0, 1, 0, 1
            
            for j in range(graph.size):
                if graph.W[j][i] != 0:
                    n1 += piT1[j] * (1 / (1 + math.e **(lambda0 - lambda1 * graph.W[i][j])))
                    n2 *= 1-piT1[j]
                    d1 += piT1[j]
                    d2 *= 1-piT1[j]
            piT2[i] = (n1 + beta * n2) / (d1 + d2)

        delta = 0
        
        for k in range(graph.size):
            delta += piT2[k] - piT1[k]
        delta /= 4
        
        for l in range(graph.size):
            piT1[l] = piT2[l]
        iter_number += 1
    print("iter_number = ", iter_number)
    return piT2


def logistic_regression(graph):
    '''
    trains a regressor to predict the sign from the links of a graph
    '''
    
    return



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

Gtest = graph(4, W, 7)

iter_max = 1000
delta_min = 0

Gtest.display

chooseParameters(Gtest, iter_max, delta_min)


