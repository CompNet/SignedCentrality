#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from igraph import *
from scipy.sparse import lil_matrix

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


def rearrange_matrix(graph):
	"""
	Rearranges the adjacency matrix of the graph to separate the individuals.

	The new matrix is such that, if the graph structure is balanced or highly balanced, the individuals would be divided in two sets.

	:param graph: graph to rearrange
	:type graph: igraph.Graph
	:return: the rearranged matrix and the sizes of its sets
	:rtype: (scipy.sparse.csr_matrix, int, int)
	"""

	matrix = get_matrix(graph)
	length = len(matrix.toarray())
	n1 = None
	n2 = None

	# TODO

	return matrix, n1, n2


def compute_eigenvector_centrality(graph):
	"""
	Compute the eigenvector centrality.

	It is computed using the method explained int the article cited in the module documentation.

	:param graph: the graph
	:type graph: igraph.Graph
	:return: the eigenvector centrality
	:rtype: list
	"""
	A, n1, n2 = rearrange_matrix(graph)  # A : valued binary symmetric matrix divided in two sets of individuals
	D = diagonal(n1, n2)  # Diagonal matrix that have the same size than A. It is divided in to sets as A too.

	B = D * A * D

	print(B)

	# TODO




