#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the modules of the package signedcentrality.
"""

import unittest
from sys import stderr
from numpy import trunc, ndarray, array
# noinspection PyProtectedMember
from signedcentrality._utils.utils import *
from signedcentrality.eigenvector_centrality import *
from signedcentrality.eigenvector_centrality import _get_matrix


class SignedCentralityTest(unittest.TestCase):

	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self._test_path_name = "test.graphml"

		samplk3 = read_graph("SAMPLK3.csv", Format.CSV)
		sampdlk = read_graph("SAMPDLK.csv", Format.CSV)

		self.graph = {
			'a': read_graph("network_a.graphml"),
			'b': read_graph("network_b.graphml"),
			's': (samplk3, sampdlk)
			}

		self.matrix = {
			'a': get_matrix(self.graph['a']),
			'b': get_matrix(self.graph['b']),
			's': _get_matrix(*self.graph['s'])
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
		self.assertSequenceEqual([round(i, 2) for i in compute_eigenvector_centrality(self.graph['a'], scaled = True)], [.43, .43, 1., .74, .74])

	def test_compute_eigenvector_centrality_network_b(self):
		self.assertSequenceEqual([trunc(i * 100) / 100 for i in compute_eigenvector_centrality(self.graph['b'], scaled = True)], [.55, .55, 1., .35, -.35])

	def test_compute_eigenvector_centrality_sampson(self):
		result = [round(i, 3) for i in compute_eigenvector_centrality(*self.graph['s'])]
		test = [.174, .188, .248, .319, .420, .219, .365, -.081, -.142, -.292, -.088, -.123, -.217, -.072, -.030, -.254, -.282, -.287]
		self.assertSequenceEqual(result, test)


if __name__ == '__main__':
	unittest.main()
