#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

	The graph that are read by this library are directed or undirected signed graphs.
	They are converted to undirected signed graphs.
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

	graph.to_undirected("collapse", dict(weight = "mean", id = "first"))  # "mean" enables the program to get the same values as the article.

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


def get_matrix(graph):
	"""
	Returns the adjacency matrix of the given graph.

	This matrix is an instance of the class scipy.sparse.csr_matrix.
	The default igraph.Matrix class isn't used because it doesn't support arithmetic operations.

	:param graph: the graph one want the adjacency matrix
	:type graph: igraph.Graph
	:return: the adjacency matrix
	:rtype: scipy.sparse.csr_matrix
	"""

	try:
		return graph.get_adjacency_sparse(FileIds.WEIGHT)  # scipy.sparse.csr_matrix
	except ValueError:
		return graph.get_adjacency_sparse()  # If there aren't weights.


