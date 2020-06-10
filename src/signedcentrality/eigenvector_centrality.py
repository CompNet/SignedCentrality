#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import float_info
from numpy import real, array
from numpy.linalg import norm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
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

	diag = lil_matrix(array([array([0. for _1 in range(n1 + n2)]) for _0 in range(n1 + n2)]))  # Create a squared matrix of floats which is initialized with the float value -1.
	diag.setdiag(-1.)
	diag.setdiag([1. for _ in range(n1)])
	return diag.tocsr()


def compute_eigenvector_centrality(graph, D=None, scaled=False):
	"""
	Compute the eigenvector centrality.

	If the graph if signed, setting the matrix D enables this function to return a centrality with the right sign.
	If this argument isn't set, the magnitude will be right, but the signs may be inverted.

	Thus, if D is set, the function solves A * x = λ * D * x instead of A * x = λ * x.
	D have to be a signed diagonal matrix.
	Values have to be set as explained in the article written by P. Bonacich and P. Lloyd.
	The function diagonal() creates such a matrix.

	:param graph: the graph
	:type graph: igraph.Graph
	:return: the eigenvector centrality
	:rtype: list
	"""

	scale = 1  # Thus, if scaled == False, the matrix won't be scaled.

	matrix = get_matrix(graph).toarray()

	eigenvector = list(        # Because eigs() returns a ndarray.
		real(                  # Because the matrix is treated as a complex matrix. So, only the real part must be kept.
			eigs(              # Because it is a square matrix.
				matrix,        # The matrix.
				1,             # To get only the first eigenvector.
				D,             # If D is set, eigs() will solve A * x = λ * D * x.
				None,          # Default value.
				"LR"           # Because the matrix is treated as a complex matrix.
				)[1]           # To get only the eigenvector (the eigenvalue is not used).
			).transpose()[0])  # Because eigs() returns a column vector.

	centrality = [value * (1 / norm(eigenvector)) for value in eigenvector]  # If the norm isn't 1, it makes the result more accurate.

	if scaled:
		max_value = float_info.min  # Minimal value of a float
		for value in centrality:
			if abs(value) > max_value:  # abs(), because the magnitude must be scaled, not the signed value.
				max_value = abs(value)
		scale = 1 / max_value
	# else, the scale remains 1.

	scaled_centrality = [value * scale for value in centrality]

	return scaled_centrality



