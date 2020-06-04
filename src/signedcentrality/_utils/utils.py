#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import igraph
from igraph import *

"""
This module contains utility functions that are used in the signedcentrality package.
"""


def readGraph(pathName):
	"""
	Read a graph from a GraphML file.

	The graph that are read by this library are undirected signed graphs.
	The function creates a Graph with the igraph library.

	:param pathName: path of the GraphML file
	:return: the Graph as an igraph.Graph
	"""
	return Graph.Read_GraphML(pathName, False, 0)

def getMatrix(graph):
	"""
	Returns the adjacency matrix of the given graph.

	This matrix is an instance of the

	:param graph: the graph one want the adjacency matrix
	:return: the adjacency matrix as an igraph.Matrix
	"""

	return graph.get_adjacency()



