#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the modules eigenvector_centrality and utils.
"""

import unittest
from igraph import Graph
from scipy.sparse import csr_matrix
from numpy import trunc, ndarray, array
from os.path import dirname, abspath, join
import sys

sys.path.insert(0, abspath(join(dirname(__file__), '../../src')))
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

# noinspection PyProtectedMember
from signedcentrality._utils.utils import write_graph, read_graph, FileIds, get_matrix, matrix_to_graph, Format
from signedcentrality.centrality.eigenvector_centrality import compute_eigenvector_centrality


def convert_graph(*args):
	"""
	Convert a graph defined in one or two files to a symmetric undirected signed graph.

	This function takes one or two parameters.
	The fist one is a graph that represents the positive signed edges.
	The second one is a graph that represents the negative signed edges.

	If there is only one graph that is set in parameters, it is considered as a positive graph.

	The graphs that are read by this function may be directed or undirected signed graphs.
	They are converted to undirected signed graphs.

	There are three steps to convert the graphs.
	First, all the weights are set to 1 in each graph.
	Secondly, The graphs are converted to undirected graphs.
	Finally, the graphs are merged.

	:param args: one or two graphs
	:type args: Graph or tuple
	:return: a symmetric undirected signed graph
	:rtype args: Graph
	"""

	if len(args) == 0 or len(args) > 2:
		raise ValueError("Wrong arguments number.")

	new_graph = None

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

		# First step :
		graph.es[FileIds.WEIGHT] = [1 for _ in range(graph.ecount())]

		# Second step :
		graph.to_undirected("collapse", dict(weight="max", id="first"))

		# Third step :
		if new_graph is None:  # If graph is the first of the list ...
			new_graph = graph
			continue  # ... the third step isn't done.

		# Else, it isn't the first one, the third step is done.
		new_matrix = get_matrix(new_graph).toarray()
		additional_matrix = get_matrix(graph).toarray()
		length = len(new_matrix)
		matrix = array([[float(new_matrix[row, col] - additional_matrix[row, col]) for col in range(length)] for row in range(length)])

		new_graph = matrix_to_graph(matrix)

	return new_graph


class SignedCentralityTest(unittest.TestCase):

	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self._test_path_name = "res/generated/test.graphml"

		samplk3 = read_graph("res/SAMPLK3.csv", Format.CSV)
		sampdlk = read_graph("res/SAMPDLK.csv", Format.CSV)

		self.graph = {
			'a': read_graph("res/network_a.graphml"),
			'b': read_graph("res/network_b.graphml"),
			's': convert_graph(samplk3, sampdlk)
		}

		self.matrix = {
			'a': get_matrix(self.graph['a']),
			'b': get_matrix(self.graph['b']),
			's': get_matrix(self.graph['s'])
		}

		self.array = {
			'a': self.matrix['a'].toarray(),
			'b': self.matrix['b'].toarray(),
			's': self.matrix['s'].toarray()
		}

	def test_get_matrix_network_a(self):
		matrix = self.matrix['a']
		array_a = self.array['a']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array_a, ndarray)

		array_test = array(
			[
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0],
				[1, 1, 0, 1, 1],
				[0, 0, 1, 0, 1],
				[0, 0, 1, 1, 0]
			])

		for i in range(len(array_a)):
			for j in range(len(array_a[i])):
				self.assertEqual(array_a[i][j], array_test[i][j])

	def test_get_matrix_network_b(self):
		matrix = self.matrix['b']
		array_b = self.array['b']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array_b, ndarray)

		array_test = array(
			[
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0],
				[1, 1, 0, 1, -1],
				[0, 0, 1, 0, 1],
				[0, 0, -1, 1, 0]
			])

		for i in range(len(array_b)):
			for j in range(len(array_b[i])):
				self.assertEqual(array_b[i][j], array_test[i][j])

	def test_get_matrix_sampson(self):
		matrix = self.matrix['s']
		array_s = self.array['s']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array_s, ndarray)

		array_test = array([
			[0, 1, 1, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1],
			[1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1],
			[0, 1, 1, 0, 1, 0, 1, 0, 0, -1, 0, 0, -1, 0, -1, -1, -1, 0],
			[1, 1, 0, 1, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
			[0, 1, 0, 0, 1, 0, 1, 0, -1, -1, 1, 0, 0, 0, 0, -1, 0, 0],
			[0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1, -1, -1],
			[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1, 0, 0, 0],
			[-1, 0, 0, 0, -1, -1, 0, 1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0],
			[0, 0, 0, -1, -1, -1, -1, 1, 0, 0, 0, 1, 1, 1, -1, -1, 1, 1],
			[0, 0, 0, 0, -1, 1, -1, 1, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0],
			[0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 1, 0, 1, 1, -1, -1, 0, 0],
			[0, 0, 0, -1, -1, 0, -1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, -1, -1, 0],
			[1, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1, 1, 0, 0, 1, 0, 1],
			[0, 0, -1, -1, -1, -1, -1, 0, 1, -1, 0, -1, 0, -1, 1, 0, 1, 1],
			[0, 0, -1, -1, -1, 0, -1, 0, 0, 1, -1, 0, 0, -1, 0, 1, 0, 1],
			[-1, 0, -1, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
		])

		for i in range(len(array_s)):
			for j in range(len(array_s[i])):
				self.assertEqual(array_s[i][j], array_test[i][j])

	def test_write_graph(self):
		graph_name = 'b'
		write_graph(self.graph[graph_name], self._test_path_name)
		test_graph = read_graph(self._test_path_name)

		array_test = test_graph.get_adjacency_sparse(FileIds.WEIGHT).toarray()

		for i in range(len(self.array[graph_name])):
			for j in range(len(self.array[graph_name][i])):
				self.assertEqual(self.array[graph_name][i][j], array_test[i][j])

	def test_compute_eigenvector_centrality_network_a(self):
		self.assertSequenceEqual([round(i, 2) for i in compute_eigenvector_centrality(self.graph['a'], scaled=True)], [.43, .43, 1., .74, .74])

	def test_compute_eigenvector_centrality_network_b(self):
		self.assertSequenceEqual([trunc(i * 100) / 100 for i in compute_eigenvector_centrality(self.graph['b'], scaled=True)], [.55, .55, 1., .35, -.35])

	def test_compute_eigenvector_centrality_sampson(self):
		result = [round(i, 3) for i in compute_eigenvector_centrality(self.graph['s'])]
		test = [.174, .188, .248, .319, .420, .219, .365, -.081, -.142, -.292, -.088, -.123, -.217, -.072, -.030, -.254, -.282, -.287]
		self.assertSequenceEqual(result, test)


if __name__ == '__main__':
	unittest.main()
