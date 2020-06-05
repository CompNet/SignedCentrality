#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy
from igraph import *
from signedcentrality.eigenvector_cenrality import *
# noinspection PyProtectedMember
from signedcentrality._utils.utils import *
from scipy.sparse import csr_matrix

"""
This module contains unit tests for the modules of the package signedcentrality.
"""


class SignedCentralityTest(unittest.TestCase):

	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self._test_path_name = "test.graphml"

		self.graph = {
			'a': read_graph("network_a.graphml"),
			'b': read_graph("network_b.graphml"),
			's': read_graph("sampson.graphml")
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
		array = self.array['a']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array, numpy.ndarray)

		array_test = numpy.array(
			[
				numpy.array([0, 0, 1, 0, 0]),
				numpy.array([0, 0, 1, 0, 0]),
				numpy.array([1, 1, 0, 1, 1]),
				numpy.array([0, 0, 1, 0, 1]),
				numpy.array([0, 0, 1, 1, 0])
				])  # Undirected graph.

		for i in range(len(array)):
			for j in range(len(array[i])):
				self.assertEqual(array[i][j], array_test[i][j])

	def test_get_matrix_network_b(self):
		matrix = self.matrix['b']
		array = self.array['b']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array, numpy.ndarray)

		array_test = numpy.array(
			[
				numpy.array([0, 0, 1, 0, 0]),
				numpy.array([0, 0, 1, 0, 0]),
				numpy.array([1, 1, 0, 1, -1]),
				numpy.array([0, 0, 1, 0, 1]),
				numpy.array([0, 0, -1, 1, 0])
				])  # Undirected graph.

		for i in range(len(array)):
			for j in range(len(array[i])):
				self.assertEqual(array[i][j], array_test[i][j])

	def test_get_matrix_sampson(self):
		matrix = self.matrix['s']
		array = self.array['s']

		self.assertIsInstance(matrix, csr_matrix)
		self.assertIsInstance(array, numpy.ndarray)

		array_test = numpy.array([
			numpy.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
			numpy.array([1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]),
			numpy.array([0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
			numpy.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
			numpy.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
			])  # Directed graph.

		for i in range(len(array)):
			for j in range(len(array[i])):
				self.assertEqual(array[i][j], array_test[i][j])

	def test_write_graph(self):
		write_graph(self.graph['s'], self._test_path_name)
		test_graph = read_graph(self._test_path_name)

		array_test = test_graph.get_adjacency_sparse(WEIGHT).toarray()

		for i in range(len(self.array['s'])):
			for j in range(len(self.array['s'][i])):
				self.assertEqual(self.array['s'][i][j], array_test[i][j])

	def test_diagonal(self):
		n1 = 4
		n2 = 4
		length = n1 + n2
		diag = diagonal(n1, n2).toarray()

		print(diag)

		array_test = numpy.array(
			[
				numpy.array([1, 0, 0, 0, 0, 0, 0, 0]),
				numpy.array([0, 1, 0, 0, 0, 0, 0, 0]),
				numpy.array([0, 0, 1, 0, 0, 0, 0, 0]),
				numpy.array([0, 0, 0, 1, 0, 0, 0, 0]),
				numpy.array([0, 0, 0, 0, -1, 0, 0, 0]),
				numpy.array([0, 0, 0, 0, 0, -1, 0, 0]),
				numpy.array([0, 0, 0, 0, 0, 0, -1, 0]),
				numpy.array([0, 0, 0, 0, 0, 0, 0, -1]),
				])

		for i in range(length):
			for j in range(length):
				self.assertEqual(diag[i][j], array_test[i][j])


if __name__ == '__main__':
	unittest.main()
