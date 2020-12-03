
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



def troll_trust(graph, beta, lambda1, iter_max, delta_min):

    print("launching troll-trust algorithm...")
    iter_number = 0

    lambda0 = -(math.log(beta/(1-beta)))
    
    piT1 = [beta for w in range(graph.taille)]
    piT2 = [beta for w in range(graph.taille)]

    delta = 100

    while (abs(delta) > delta_min):

        for i in range(graph.taille):
            
            n1, n2, d1, d2 = 0, 1, 0, 1
            
            for j in range(graph.taille):
                if graph.W[j][i] != 0:
                    n1 += piT1[j] * (1 / (1 + math.e **(lambda0 - lambda1 * graph.W[i][j])))
                    n2 *= 1-piT1[j]
                    d1 += piT1[j]
                    d2 *= 1-piT1[j]
            piT2[i] = (n1 + beta * n2) / (d1 + d2)

        delta = 0
        
        for k in range(graph.taille):
            delta += piT2[k] - piT1[k]
        delta /= 4

        
        for l in range(graph.taille):
            piT1[l] = piT2[l]
        iter_number += 1
    print("iter_number = ", iter_number)
    return piT2


# MAIN:
''' values : '''
W = [[0, 0, -0.1, 0.1],
     [1.0, 0, -0.9, 0],
     [0, -0.6, 0, 0],
     [0.8, 0, 0.7, 0]]
Gtest = graph(4, W)

iter_max = 1000
delta_min = 0.000000000000001


print("#############################################################")
print("#############################################################")
print("#########                                           #########")
print("#########                 Troll-trust               #########")
print("#########                                           #########")
print("#############################################################")
print("#############################################################")

Gtest.display
'''
pi = troll_trust(Gtest, beta, lambda1, iter_max)
for c1 in range(Gtest.taille):
    print(c1, " => ", pi[c1])
'''

for i in numpy.arange(0.01, 1, 0.01):
    lambda1 = i
    for j in numpy.arange(0.01, 1, 0.01):
        beta = j
        pi = troll_trust(Gtest, beta, lambda1, iter_max, delta_min)
        print("lambda1 = ", lambda1, "beta = ", beta)
        print("pi:")
        for c1 in range(Gtest.taille):
            print(c1, " => ", pi[c1])
        if((0.6 <= pi[0] < 0.7) and (0.5 <= pi[1] < 0.6) and (0.4 <= pi[2] < 0.5) and (0.5 <= pi[3] < 0.6)):
            sys.exit("Approximation found!")
        print("\n")

