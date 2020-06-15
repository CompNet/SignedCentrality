#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import float_info
from numpy import real, where, array
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from signedcentrality._utils.utils import *
from signedcentrality._utils import FileIds

"""
This module contains functions related to the measure of eigenvector centrality.

The measure is computed by following the method of Phillip Bonacich and Paulette Lloyd.

.. note: P. Bonacich & P. Lloyd. (2004). Calculating status with negative relations. SocialNetworks, 26, 331-338. 10.1016/j.socnet.2004.08.007
"""


def _set_weights_to_1(graph):
	"""
	Set all weights in a graph to 1

	This function shouldn't be used outside this module.

	:param graph: the graph whose the weights are to be changed
	:type graph: Graph
	:return: the new graph
	:rtype graph: Graph
	"""

	graph.es[FileIds.WEIGHT] = [1 for _ in range(graph.ecount())]
	return graph


def _get_matrix(*args):
	"""
	Convert a graph defined in one or two files to a symmetric undirected signed graph.

	This function takes one or two parameters.
	The fist one is a graph that represents the positive signed edges.
	The second one is a graph that represents the negative signed edges.

	If there is only one graph that is set in parameters, it is considered as a positive graph.

	The graphs that are read by this library may be directed or undirected signed graphs.
	They are converted to undirected signed graphs.

	There are three steps to convert the graphs.
	First, all the weights are set to 1 in each graph.
	Secondly, The graphs are converted to undirected graphs.
	Finally, the two graphs are merged.

	This function shouldn't be used outside this module.

	:param args: on or two graphs
	:type args: Graph or tuple
	:return: a symmetric undirected signed graph
	:rtype args: Graph
	"""

	if len(args) == 0 or len(args) > 2:
		raise ValueError("Wrong arguments number.")

	new_graph = None
	matrix = None

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

		# First step :
		graph = _set_weights_to_1(graph)

		# Second step :
		graph.to_undirected("collapse", dict(weight = "max", id = "first"))  # "mean" enables the program to get the same values as the article.

		# Third step :
		if new_graph is None:  # If graph is the first of the list ...
			new_graph = graph
			continue  # ... the third step isn't done.

		# Else, it isn't the first one, the third step is done.
		new_matrix = get_matrix(new_graph).toarray()
		additional_matrix = get_matrix(graph).toarray()
		length = len(new_matrix)
		matrix = array([[float(new_matrix[row, col] - additional_matrix[row, col]) for col in range(length)] for row in range(length)])

	return csr_matrix(matrix)


def compute_eigenvector_centrality(*graph, scaled=False):
	"""
	Compute the eigenvector centrality.

	If scaled is True, the values will be set such that the maximum is 1.
	This argument name have to be set in the function call.

	The graph must be an undirected signed graph or two unsigned graphs.
	If there are two graphs, the first one represent the positive weights and the second one defines the negative edges.

	:param graph: the graph
	:type graph: igraph.Graph or tuple
	:param scaled: the graph
	:type scaled: bool
	:return: the eigenvector centrality
	:rtype: list
	"""

	if len(graph) == 0 or len(graph) > 2:
		raise ValueError("Wrong arguments number.")

	scale = 1  # Thus, if scaled == False, the matrix won't be scaled.
	matrix = None

	if len(graph) == 1:
		matrix = get_matrix(graph[0]).toarray()
	else:  # if len(graph) == 2
		matrix = _get_matrix(*graph).toarray()

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
		max_value = float_info.min  # Minimal value of a float
		for value in centrality:
			if abs(value) > max_value:  # abs(), because the magnitude must be scaled, not the signed value.
				max_value = abs(value)
		scale = 1 / max_value
	# else, the scale remains 1.

	if sum(centrality) < 0:  # Makes the first cluster values positive if they aren't.
		scale *= -1  # Values will be inverted when they will be scaled (more efficient).

	if scale == 1:  # If the centrality has the right signs and if it doesn't have to be scaled, it can be returned.
		return centrality

	return [value * scale for value in centrality]  # Else, return a scaled centrality.



