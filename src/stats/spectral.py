'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import scipy.sparse.linalg
import numpy as np

def retreive_largest_eigenvalue(g):
    """This method retrieves the largest eigenvalue of a given signed graph.
       
    :param g: graph
    :type g: igraph.Graph
    :return: the largest eigenvalue
    :rtype: float
    """
    matrix = util.get_matrix(g).toarray()
    res = scipy.sparse.linalg.eigs(matrix)
    eigenvalues = res[0]
    #eigenvectors = res[1] # eigenvectors[:, 2] is the 3rd eigenvector
    return np.real(max(eigenvalues))