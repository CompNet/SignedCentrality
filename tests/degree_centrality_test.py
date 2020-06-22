#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the modules degree_centrality.
"""

import unittest
from signedcentrality import degree_centrality
from csv import reader, Sniffer
from scipy.sparse import csr_matrix
from numpy import trunc, ndarray, array, transpose
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
		gamaneg = read_CSV("GAMANEG.csv")
		gamaboth = convert_graph(gamapos, gamaneg)

		samplk3 = read_CSV("SAMPLK3.csv")
		sampdlk = read_CSV("SAMPDLK.csv")
		sampson = convert_graph(samplk3, sampdlk)

		self.graph = {
			'gamapos': gamapos,
			'gamaneg': gamaneg,
			'gamaboth': gamaboth,
			'samplk3': samplk3,
			'sampdlk': sampdlk,
			'sampson': sampson,
			'5': read_CSV("table_5.csv")
			}

		self.matrix = {
			'gamapos': get_matrix(self.graph['gamapos']),
			'gamaneg': get_matrix(self.graph['gamaneg']),
			'gamaboth': get_matrix(self.graph['gamaboth']),
			'samplk3': get_matrix(self.graph['samplk3']),
			'sampdlk': get_matrix(self.graph['sampdlk']),
			'sampson': get_matrix(self.graph['sampson']),
			'5': get_matrix(self.graph['5'])
			}

		self.array = {
			'gamapos': self.matrix['gamapos'].toarray(),
			'gamaneg': self.matrix['gamaneg'].toarray(),
			'gamaboth': self.matrix['gamaboth'].toarray(),
			'samplk3': self.matrix['samplk3'].toarray(),
			'sampdlk': self.matrix['sampdlk'].toarray(),
			'sampson': self.matrix['sampson'].toarray(),
			'5': self.matrix['5'].toarray()
			}

	def test_read_graph(self):
		array_test = array([
			[0., 1., 0., 1., 0., 0., 0., -1., -1., 0.],
			[1., 0., 1., -1., 1., -1., -1., 0., 0., 0.],
			[0., 1., 0., 1., -1., 0., 0., 0., -1., 0.],
			[1., -1., 1., 0., 1., -1., -1., 0., 0., 0.],
			[0., 1., -1., 1., 0., 1., 0., -1., 0., -1.],
			[0., -1., 0., -1., 1., 0., 1., 0., 1., -1.],
			[0., -1., 0., -1., 0., 1., 0., 1., -1., 1.],
			[- 1., 0., 0., 0., -1., 0., 1., 0., 1., 0.],
			[- 1., 0., -1., 0., 0., 1., -1., 1., 0., 1.],
			[0., 0., 0., 0., -1., -1., 1., 0., 1., 0.]
			])

		for i in range(min(len(array_test), len(self.array['5']))):
			for j in range(min(len(array_test[i]), len(self.array['5'][i]))):
				self.assertEqual(self.array['5'][i][j], array_test[i][j])

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
		graph.to_directed()
		graph.add_edge(1, 0)
		graph.add_edge(2, 0)
		graph.add_edge(3, 0)
		graph.add_edge(4, 0)
		print(get_matrix(graph).toarray())
		print()

		undirected_graph = Graph(5)
		undirected_graph.to_undirected()
		undirected_graph.add_edge(1, 0)
		undirected_graph.add_edge(2, 0)
		undirected_graph.add_edge(3, 0)
		undirected_graph.add_edge(4, 0)
		print(get_matrix(undirected_graph).toarray())
		print()

		result = [round(x, 1) for x in degree_centrality.PositiveCentrality.undirected(undirected_graph)]
		print("P undirected :", result)
		result_in = [round(x, 1) for x in degree_centrality.PositiveCentrality.incoming(graph)]
		print("P incoming :  ", result_in)
		result_out = [round(x, 1) for x in degree_centrality.PositiveCentrality.outgoing(graph)]
		print("P outgoing :  ", result_out)
		print()

		result = [round(x, 1) for x in degree_centrality.NegativeCentrality.undirected(undirected_graph)]
		print("N undirected :", result)
		result_in = [round(x, 1) for x in degree_centrality.NegativeCentrality.incoming(graph)]
		print("N incoming :  ", result_in)
		result_out = [round(x, 1) for x in degree_centrality.NegativeCentrality.outgoing(graph)]
		print("N outgoing :  ", result_out)

		self.assertSequenceEqual(result_in, test_in)
		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected(self):

		test = [1.030, 1.157, 1.074, 0.897, 0.775, 0.812, 0.763, 0.942, 0.900, 0.993, 0.853, 0.952, 0.860, 0.855, 0.645, 0.393, 0.476, 0.415]
		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['sampson'])]

		test_in = [0.000, 1.000, 1.000, 1.000, 1.000]
		test_out = [1.000, 0.996, 0.996, 0.996, 0.996]

		print("test :      ", test)
		print("undirected :", result)
		result_in = [round(x, 1) for x in degree_centrality.PNCentrality.incoming(self.graph['sampson'])]
		print("incoming :  ", result_in)
		result_out = [round(x, 1) for x in degree_centrality.PNCentrality.outgoing(self.graph['sampson'])]
		print("outgoing :  ", result_out)

		print()

		print(self.array['5'])

		test = [0.901, 0.861, 0.908, 0.861, 0.841, 0.850, 0.862, 0.902, 0.851, 0.907]
		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['5'])]
		result_in = [round(x, 3) for x in degree_centrality.PNCentrality.incoming(self.graph['5'])]
		result_out = [round(x, 3) for x in degree_centrality.PNCentrality.outgoing(self.graph['5'])]

		print(test)
		print(result)
		print(result_out)
		print(result_in)

		self.assertSequenceEqual(result, test)
		self.assertSequenceEqual(result_in, test)
		self.assertSequenceEqual(result_out, test)

		self.assertSequenceEqual(test, result)


if __name__ == '__main__':
	unittest.main()
