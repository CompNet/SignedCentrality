#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the modules degree_centrality.
"""

import unittest
from signedcentrality import degree_centrality
from csv import reader, Sniffer
from scipy.sparse import csr_matrix
from numpy import trunc, ndarray, array
from igraph import Graph
from signedcentrality._utils.utils import *


def read_CSV(path):
	"""
	Creates an igraph.Graph from a CSV file

	:param path: the path of the CSV file
	:type path: str
	:return: the graph
	:rtype: igraph.Graph
	"""

	matrix = None
	csv = []

	with open(path) as file:

		dialect = Sniffer().sniff(file.read(1024))
		file.seek(0)

		header = Sniffer().has_header(file.read(1024))
		file.seek(0)

		for row in reader(file, dialect):
			csv.append(row)

		matrix = array([[float(csv[i][j]) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))])  # int(header) is 0 if False and 1 if true

	return matrix_to_graph(array(matrix))


def convert_graph(*args, directed = True):
	"""
	Convert a graph defined in one or two files to a symmetric undirected signed graph.

	This function takes two or three parameters.
	The fist one is a graph that represents the positive signed edges.
	The optional second one is a graph that represents the negative signed edges.
	The third one indicates if the merged graph has to be directed

	If there is only one graph that is set in parameters, it is considered as a positive graph.
	The graphs that are read by this function may be directed or undirected signed graphs.

	:param args: one or two graphs
	:type args: igraph.Graph or tuple
	:param directed: indicates if the merged graph has to be directed
	:type directed: Bool
	:return: the merged graph
	:rtype args: igraph.Graph
	"""

	if len(args) == 0 or len(args) > 2:
		raise ValueError("Wrong arguments number.")

	new_graph = None

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

		if not directed:
			graph.to_undirected("collapse")

		if new_graph is None:  # If graph is the first of the list ...
			new_graph = graph
			continue  # ... the third step isn't done.

		# Else, it isn't the first one.
		new_matrix = get_matrix(new_graph).toarray()
		additional_matrix = get_matrix(graph).toarray()
		length = len(new_matrix)
		matrix = array([[float(new_matrix[row, col] - additional_matrix[row, col]) for col in range(length)] for row in range(length)])

		new_graph = matrix_to_graph(matrix)

	return new_graph


class DegreeCentralityTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		gamapos = read_CSV("GAMAPOS.csv")
		# print(get_matrix(gamapos).toarray());print()
		gamaneg = read_CSV("GAMANEG.csv")
		# print(get_matrix(gamaneg).toarray());print()
		gamaboth = convert_graph(gamapos, gamaneg)
		# print(get_matrix(graph).toarray())

		self.graph = {
			'gamapos': gamapos,
			'gamaneg': gamaneg,
			'gamaboth': gamaboth
			}

		self.matrix = {
			'gamapos': get_matrix(self.graph['gamapos']),
			'gamaneg': get_matrix(self.graph['gamaneg']),
			'gamaboth': get_matrix(self.graph['gamaboth'])
			}

		self.array = {
			'gamapos': self.matrix['gamapos'].toarray(),
			'gamaneg': self.matrix['gamaneg'].toarray(),
			'gamaboth': self.matrix['gamaboth'].toarray()
			}

	def test_positive_centrality_undirected(self):
		# print(self.array['gamaneg'])
		# print(self.graph['gamaneg'])

		# test = [71.83, 72.46, 90.38, 95.21, 81.31, 75.05, 100.00, 94.41, 80.69, 85.69, 74.44, 80.56, 80.97, 83.78, 67.76, 68.26]
		# # result = [round(100 * x, 2) for x in degree_centrality.PositiveCentrality.undirected(self.graph['gamapos'], True)]
		# # self.graph['gamapos'].to_undirected("collapse")
		# result = [round(100 * x, 2) for x in degree_centrality.NegativeCentrality.undirected(self.graph['gamaneg'], True)]
		# # result = [100 * x for x in degree_centrality.NegativeCentrality.undirected(self.graph['gamapos'], True)]
		# # result.sort()
		# # test.sort()

		test_in = [0.000, 1.000, 1.000, 1.000, 1.000]
		test_out = [1.000, 0.996, 0.996, 0.996, 0.996]

		graph = Graph(5)
		graph.to_undirected()
		graph.add_edge(1, 0)
		graph.add_edge(2, 0)
		graph.add_edge(3, 0)
		graph.add_edge(4, 0)

		print("test in :   ", test_in)
		print()

		result = [round(x, 1) for x in degree_centrality.NegativeCentrality.undirected(graph)]
		# result = [round(x, 1) for x in degree_centrality.PositiveCentrality.undirected(graph)]
		print("undirected :", result)
		result_in = [round(x, 1) for x in degree_centrality.NegativeCentrality.incoming(graph)]
		# result_in = [round(x, 1) for x in degree_centrality.PositiveCentrality.incoming(graph)]
		print("incoming :  ", result_in)
		result_out = [round(x, 1) for x in degree_centrality.NegativeCentrality.outgoing(graph)]
		# result_out = [round(x, 1) for x in degree_centrality.PositiveCentrality.outgoing(graph)]
		print("outgoing :  ", result_out)

		self.assertSequenceEqual(result_in, test_in)
		self.assertSequenceEqual(result_out, test_out)


if __name__ == '__main__':
	unittest.main()
