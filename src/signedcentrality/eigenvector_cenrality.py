#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from igraph import *
from numpy import *
from scipy.sparse import lil_matrix
from scipy.linalg import eigh

from signedcentrality._utils.utils import *

"""
This module contains functions related to the measure of eigenvector centrality.

The measure is computed by following the method of Phillip Bonacich and Paulette Lloyd.

.. note: P. Bonacich & P. Lloyd. (2004). Calculating status with negative relations. SocialNetworks, 26, 331-338. 10.1016/j.socnet.2004.08.007

"""


def diagonal(n1, n2):
	"""
	Create a diagonal squared matrix.

	Values are set as explained in the article written by P. Bonacich and P. Lloyd.
	Using scipy.sparse.lil_matrix is more efficient to contruct the matrix.
	Since it is to slow to compute arithmetic operations, it is converted in csr_matrix which is more efficient to such operations.

	:param n1: number of individuals in the first set
	:type n1: int
	:param n2: number of individuals in the second set
	:type n2: int
	:return: the matrix
	:rtype: scipy.sparse.csr_matrix
	"""

	diag = lil_matrix(numpy.array([numpy.array([0. for _1 in range(n1 + n2)]) for _0 in range(n1 + n2)]))  # Create a squared matrix of floats which is initialized with the float value -1.
	diag.setdiag(-1.)
	diag.setdiag([1. for _ in range(n1)])
	return diag.tocsr()


def compute_eigenvector_centrality(graph):
	"""
	Compute the eigenvector centrality.

	It is computed using the method explained int the article cited in the module documentation.

	:param graph: the graph
	:type graph: igraph.Graph
	:return: the eigenvector centrality
	:rtype: list
	"""

	matrix = get_matrix(graph).toarray()

	eigenvalues, eigenvectors = eigh(matrix)

	D = eye(len(eigenvectors))
	print(D)
	print()
	print(matrix)
	print()
	print(eigenvalues)
	print()
	print(eigenvectors)

	for i in range(5):

		print()
		print("1 =>", graph.evcent(False, True, WEIGHT))  # Works only for unsigned graphs.
		print("1 =>", graph.evcent(False, False, WEIGHT))

		print()
		Dx = eigenvectors[:, i]
		print("2 =>", dot(eigenvalues[i], Dx))

		print()
		print("3 =>", dot(matrix, Dx))

		print()
		print("4 =>", Dx)

		print()
		print("4 =>", linalg.norm(eigenvectors[:, i]))

		print()

	return Dx  # temporary




