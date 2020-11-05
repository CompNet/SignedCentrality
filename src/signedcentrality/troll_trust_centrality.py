
import math
import numpy
import sys

class graph:   
    
    def __init__(self, taille, W):
        self.taille = taille
        self.W = W

    def display(self):
        print("\n matrix values:")
        for i in range(self.taille):
            for j in range(self.taille):
                print(self.W[i][j], "  ")
            print('\n')



def troll_trust(graph, beta, lambda1, limit):
'''
returns the list "pi" containing the centrality values for each node
'''
    print("launching troll-trust algorithm...")
    t = 0
    pi = [0 for x in range(graph.taille)]
    for y in range (graph.taille):
        pi[y] = beta
    lambda0 = -(math.log(beta/(1-beta)))
    while t != limit:
        n1, n2, d1, d2 = 0, 1, 0, 1
        for i in range(graph.taille):
            for j in range(graph.taille):
                if graph.W[j][i] != 0:
                    n1 += pi[j] * (1 / (1 + math.e **(lambda0-lambda1) * graph.W[i][j]))
                    n2 *= 1-pi[j]
                    d1 += pi[j]
                    d2 *= 1-pi[j]
                    pi[i] = (n1 + beta * n2) / (d1 + d2)
        t += 1
    return pi


# MAIN:
''' values : '''
W = [[0, 0, -0.1, 0.1],
     [1.0, 0, -0.9, 0],
     [0, -0.6, 0, 0],
     [0.8, 0, 0.7, 0]]
Gtest = graph(4, W)

limit = 50


print("#############################################################")
print("#############################################################")
print("#########                                           #########")
print("#########                 Troll-trust               #########")
print("#########                                           #########")
print("#############################################################")
print("#############################################################")

Gtest.display
'''
pi = troll_trust(Gtest, beta, lambda1, limit)
for c1 in range(Gtest.taille):
    print(c1, " => ", pi[c1])
'''

for i in numpy.arange(0.025, 1, 0.001):
    lambda1 = i
    for j in numpy.arange(0.25, 1, 0.001):
        beta = j
        pi = troll_trust(Gtest, beta, lambda1, limit)
        print("lambda1 = ", lambda1, "beta = ", beta)
        print("pi:")
        for c1 in range(Gtest.taille):
            print(c1, " => ", pi[c1])
        if((0.6 <= pi[0] < 0.7) and (0.5 <= pi[1] < 0.6) and (0.4 <= pi[2] < 0.5) and (0.5 <= pi[3] < 0.6)):
            sys.exit("Approximation found!")
        print("\n")

