#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import float_info
from igraph import Graph
from signedcentrality._utils import *  # Import the strings defined in __init__.py.

"""
This module contains utility functions that are used in the signedcentrality package.
"""


def read_graph(path_name, format = None):
	"""
	Read a graph from a file.

	It can read some file formats, such as GraphML or CSV.
	XML-like files such as GraphML have to be written in a standard format.
	See example below for GraphML files.

	If format is None, it will be detected automatically.
	It might cause errors. It is preferred that the format has been set.

	The function creates a Graph with the igraph library.

	:param path_name: path of the file
	:type path_name: str
	:param format: format of the file
	:type format: str
	:return: the graph
	:rtype: igraph.Graph

	Here is an example of how the GraphML file has to be written.
	This GraphML file uses the standards of the igraph.graph.write_graphml() method.

	:Example:

	<?xml version="1.0" encoding="UTF-8"?>
	<graphml
			xmlns="http://graphml.graphdrawing.org/xmlns"
			xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
			xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
			http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
		<key id="v_id" for="node" attr.name="id" attr.type="string"/>
		<key id="e_weight" for="edge" attr.name="weight" attr.type="double"/>
		<key id="e_id" for="edge" attr.name="id" attr.type="string"/>
		<graph id="network_a" edgedefault="undirected">
			<node id="n0">
			<data key="v_id">1</data>
			</node>
			<node id="n1">
			<data key="v_id">2</data>
			</node>
			<node id="n2">
			<data key="v_id">3</data>
			</node>
			<edge source="n0" target="n1">
			<data key="e_weight">1</data>
			<data key="e_id">e0</data>
			</edge>
			<edge source="n1" target="n2">
			<data key="e_weight">1</data>
			<data key="e_id">e1</data>
			</edge>
			<edge source="n2" target="n3">
			<data key="e_weight">1</data>
			<data key="e_id">e2</data>
			</edge>
		</graph>
	</graphml>

	.. warning: In the GraphML file, the attribute which defines the weights of the edges has to be given the name "weight" with the attribute "attr.name".
	"""

	graph = None

	if format is not None and format.lower() == Format.CSV:
		graph = Graph.Read_Adjacency(path_name, ",", "#", FileIds.WEIGHT)  # The separator in CSV files is the comma.
	else:
		graph = Graph.Read(path_name, format)

	return graph


def write_graph(graph, path_name):
	"""
	Write a GraphML file from a Graph.

	This function is used for tests.

	:param graph: graph to write
	:type graph: igraph.Graph
	:param path_name: path of the GraphML file
	:type path_name: str
	"""

	graph.write_graphml(path_name)


def get_matrix(graph, weights = FileIds.WEIGHT):
	"""
	Returns the adjacency matrix of the given graph.

	This matrix is an instance of the class scipy.sparse.csr_matrix.
	The default igraph.Matrix class isn't used because it doesn't support arithmetic operations.

	:param weights: name of the weights
	:type weights: str
	:param graph: the graph one want the adjacency matrix
	:type graph: igraph.Graph
	:return: the adjacency matrix
	:rtype: scipy.sparse.csr_matrix
	"""

	try:
		return graph.get_adjacency_sparse(weights)  # scipy.sparse.csr_matrix
	except ValueError:
		return graph.get_adjacency_sparse()  # If there aren't weights.


def get_scale(centrality, fit_sign = False):
	"""
	Compute the scale value to scale a centrality

	If fit_sign is True, the sign of the scale is changed when the sum of the centrality is negative.
	It is used to make the first clique positive.

	:param centrality: the centrality which the scale is to be computed
	:type centrality: numpy.ndarray
	:param fit_sign: indicates if the sign must be changed
	:type fit_sign: bool
	:return: the scale
	:rtype: float
	"""

	max_value = float_info.min  # Minimal value of a float
	for value in centrality:
		if abs(value) > max_value:  # abs(), because the magnitude must be scaled, not the signed value.
			max_value = abs(value)
	scale = 1 / max_value

	if fit_sign and sum(centrality) < 0:  # Makes the first cluster values positive if they aren't.
		scale *= -1  # Values will be inverted when they will be scaled (more efficient).

	return scale


def scale_centrality(centrality, fit_sign = False):
	"""
	Scale the given centrality

	If fit_sign is True, the sign of the scale is changed when the sum of the centrality is negative.
	It is used to make the first clique positive.

	:param centrality: the centrality which the scale is to be computed
	:type centrality: numpy.ndarray
	:param fit_sign: indicates if the sign must be changed
	:type fit_sign: bool
	:return: the scaled centrality
	:rtype: numpy.ndarray
	"""

	scale_ = get_scale(centrality, fit_sign)

	if scale_ == 1:  # If the centrality has the right signs and if it doesn't have to be scaled, it can be returned.
		return centrality

	return [value * scale_ for value in centrality]  # Else, return a scaled centrality.


def matrix_to_graph(matrix, weight_attr = FileIds.WEIGHT):
	"""
	Creates a graph from a numpy adjacency matrix

	:param matrix: the adjacency matrix
	:type matrix: numpy.ndarray
	:param weight_attr: attribute name to use for weights
	:type weight_attr: str
	:return: the graph
	:rtype: igraph.Graph
	"""

	length = len(matrix)
	graph = Graph()
	graph.to_directed()  # If an undirected graph is needed, it will be done later.
	graph.add_vertices(length)

	for row in range(length):
		for col in range(length):
			weight = matrix[row, col]
			if weight != 0:
				if weight_attr == FileIds.SIGN:
					graph.add_edge(row, col, sign = weight)
				else:  # if weight_attr == FileIds.WEIGHT or it is a wrong value
					graph.add_edge(row, col, weight = weight)

	return graph

