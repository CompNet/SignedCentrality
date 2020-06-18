#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import float_info
from numpy import real, where, array
from numpy.linalg import norm
from scipy.sparse.linalg import eigs
from signedcentrality._utils.utils import *

"""
This module contains functions related to the measure of eigenvector centrality.

The measure is computed by following the method of Phillip Bonacich and Paulette Lloyd.

.. note: P. Bonacich & P. Lloyd. (2004). Calculating status with negative relations. SocialNetworks, 26, 331-338. 10.1016/j.socnet.2004.08.007
"""


def compute_eigenvector_centrality(graph, scaled=False):
	"""
	Compute the eigenvector centrality.

	If scaled is True, the values will be set such that the maximum is 1.

	The graph must be an undirected signed graph or two unsigned graphs.
	If there are two graphs, the first one represent the positive weights and the second one defines the negative edges.

	:param graph: the graph
	:type graph: igraph.Graph or tuple
	:param scaled: indicates if the centrality must be scaled
	:type scaled: bool
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
				None,          # Default value.
				None,          # Default value.
				"LR"           # Because the matrix is treated as a complex matrix.
				)[1]           # To get only the eigenvector (the eigenvalue is not used).
			).transpose()[0])  # Because eigs() returns a column vector.

	centrality = [value * (1 / norm(eigenvector)) for value in eigenvector]  # If the norm isn't 1, it makes the result more accurate.

	if scaled:  # Sets the values such that the maximum is 1
		scale = get_scale(centrality)
	# else, the scale remains 1.

	if sum(centrality) < 0:  # Makes the first cluster values positive if they aren't.
		scale *= -1  # Values will be inverted when they will be scaled (more efficient).

	if scale == 1:  # If the centrality has the right signs and if it doesn't have to be scaled, it can be returned.
		return centrality

	return [value * scale for value in centrality]  # Else, return a scaled centrality.



