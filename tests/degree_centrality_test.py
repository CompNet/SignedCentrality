#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the modules degree_centrality.
"""

import unittest
from signedcentrality import degree_centrality
from csv import reader, Sniffer
from scipy.sparse import csr_matrix
from numpy import trunc, ndarray, array, transpose, triu, tril
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
	g = None

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

		if not directed:
			g = graph.as_undirected("collapse")
		else:
			g = graph

		if new_graph is None:  # If graph is the first of the list ...
			# new_graph = graph
			new_graph = g
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

		graph_2_directed = Graph(5)
		graph_2_directed.to_directed()
		graph_2_directed.add_edge(1, 0)
		graph_2_directed.add_edge(2, 0)
		graph_2_directed.add_edge(3, 0)
		graph_2_directed.add_edge(4, 0)
		# print(get_matrix(graph_2_directed).toarray())
		# print()

		graph_2_undirected = Graph(5)
		graph_2_undirected.to_undirected()
		graph_2_undirected.add_edge(1, 0)
		graph_2_undirected.add_edge(2, 0)
		graph_2_undirected.add_edge(3, 0)
		graph_2_undirected.add_edge(4, 0)
		# print(get_matrix(graph_2_undirected).toarray())
		# print()

		graph_5_directed = read_CSV("table_5.csv")
		graph_5_undirected = read_CSV("table_5.csv")
		graph_5_undirected.to_undirected("collapse", dict(weight = "mean", id = "first"))


		self.graph = {
			'gamapos': gamapos,
			'gamaneg': gamaneg,
			'gamaboth': gamaboth,
			'samplk3': samplk3,
			'sampdlk': sampdlk,
			'sampson': sampson,
			'2_directed': graph_2_directed,
			'2_undirected': graph_2_undirected,
			'5_directed': graph_5_directed,
			'5_undirected': graph_5_undirected
			}

		self.matrix = {
			'gamapos': get_matrix(self.graph['gamapos']),
			'gamaneg': get_matrix(self.graph['gamaneg']),
			'gamaboth': get_matrix(self.graph['gamaboth']),
			'samplk3': get_matrix(self.graph['samplk3']),
			'sampdlk': get_matrix(self.graph['sampdlk']),
			'sampson': get_matrix(self.graph['sampson']),
			'2_directed': get_matrix(self.graph['2_directed']),
			'2_undirected': get_matrix(self.graph['2_undirected']),
			'5_directed': get_matrix(self.graph['5_directed']),
			'5_undirected': get_matrix(self.graph['5_undirected'])
			}

		self.array = {
			'gamapos': self.matrix['gamapos'].toarray(),
			'gamaneg': self.matrix['gamaneg'].toarray(),
			'gamaboth': self.matrix['gamaboth'].toarray(),
			'samplk3': self.matrix['samplk3'].toarray(),
			'sampdlk': self.matrix['sampdlk'].toarray(),
			'sampson': self.matrix['sampson'].toarray(),
			'2_directed': self.matrix['2_directed'].toarray(),
			'2_undirected': self.matrix['2_undirected'].toarray(),
			'5_directed': self.matrix['5_directed'].toarray(),
			'5_undirected': self.matrix['5_undirected'].toarray()
			}

	# def test_read_graph(self):
	# 	array_test = array([
	# 		[0., 1., 0., 1., 0., 0., 0., -1., -1., 0.],
	# 		[1., 0., 1., -1., 1., -1., -1., 0., 0., 0.],
	# 		[0., 1., 0., 1., -1., 0., 0., 0., -1., 0.],
	# 		[1., -1., 1., 0., 1., -1., -1., 0., 0., 0.],
	# 		[0., 1., -1., 1., 0., 1., 0., -1., 0., -1.],
	# 		[0., -1., 0., -1., 1., 0., 1., 0., 1., -1.],
	# 		[0., -1., 0., -1., 0., 1., 0., 1., -1., 1.],
	# 		[- 1., 0., 0., 0., -1., 0., 1., 0., 1., 0.],
	# 		[- 1., 0., -1., 0., 0., 1., -1., 1., 0., 1.],
	# 		[0., 0., 0., 0., -1., -1., 1., 0., 1., 0.]
	# 		])
	#
	# 	for i in range(min(len(array_test), len(self.array['5_directed']))):
	# 		for j in range(min(len(array_test[i]), len(self.array['5_directed'][i]))):
	# 			self.assertEqual(self.array['5_directed'][i][j], array_test[i][j])

	def test_positive_centrality_undirected_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		# print(self.array['gamapos'])

		test_undirected = [1.111111, 1.111111, 1.159564, 1.079804, 1.115326, 1.199719, 1.272832, 1.234552, 1.111397, 1.075419, 1.162314, 1.162314, 1.151173, 1.075550, 1.111111, 1.111111]
		test_in = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]
		test_out = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]

		result = [round(100 * x, 6) for x in degree_centrality.PositiveCentrality.undirected(self.graph['gamapos'])]

		print("test undirected : ", test_undirected)
		print("result :          ", result)
		print()

		print("test in :         ", test_in)
		print("test out :        ", test_out)

		self.assertSequenceEqual(result, test_undirected)

	def test_positive_centrality_in_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		# print(self.array['gamapos'])

		test_undirected = [1.111111, 1.111111, 1.159564, 1.079804, 1.115326, 1.199719, 1.272832, 1.234552, 1.111397, 1.075419, 1.162314, 1.162314, 1.151173, 1.075550, 1.111111, 1.111111]
		test_in = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]
		test_out = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]

		result = [round(100 * x, 6) for x in degree_centrality.PositiveCentrality.incoming(self.graph['gamapos'])]

		print("test in :         ", test_in)
		print("result :          ", result)
		print()

		print("test undirected : ", test_undirected)
		print("test out :        ", test_out)

		self.assertSequenceEqual(result, test_in)

	def test_positive_centrality_out_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		# print(self.array['gamapos'])

		test_undirected = [1.111111, 1.111111, 1.159564, 1.079804, 1.115326, 1.199719, 1.272832, 1.234552, 1.111397, 1.075419, 1.162314, 1.162314, 1.151173, 1.075550, 1.111111, 1.111111]
		test_in = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]
		test_out = [0.9182736, 0.9182736, 0.9057796, 0.9528579, 0.9239120, 0.8818301, 0.8249723, 0.8501708, 0.9179550, 0.9474681, 0.9095186, 0.9095186, 0.8945808, 0.9472861, 0.9182736, 0.9182736]

		result = [round(100 * x, 6) for x in degree_centrality.PositiveCentrality.outgoing(self.graph['gamapos'])]

		print("test out :        ", test_out)
		print("result :          ", result)
		print()

		print("test undirected : ", test_undirected)
		print("test in :         ", test_in)

		self.assertSequenceEqual(result, test_out)

	def test_negative_centrality_undirected_gamaneg(self):  # Works correctly
		print(self.array['gamaneg'])
		print(self.graph['gamaneg'])

		test = [71.83, 72.46, 90.38, 95.21, 81.31, 75.05, 100.00, 94.41, 80.69, 85.69, 74.44, 80.56, 80.97, 83.78, 67.76, 68.26]
		result = [round(100 * x, 2) for x in degree_centrality.NegativeCentrality.undirected(self.graph['gamaneg'], True)]

		print(test)
		print(result)

		self.assertSequenceEqual(result, test)

	# def test_positive_centrality_in_2(self):
	# 	# test_in = [0.000, 1.000, 1.000, 1.000, 1.000]  # Not the right values ...
	# 	#
	# 	# result = [round(x, 1) for x in degree_centrality.PositiveCentrality.undirected(self.graph['2_undirected'])]
	# 	# print("P undirected :", result)
	# 	# result_in = [round(x, 1) for x in degree_centrality.PositiveCentrality.incoming(self.graph['2_directed'])]
	# 	# print("P incoming :  ", result_in)
	# 	# result_out = [round(x, 1) for x in degree_centrality.PositiveCentrality.outgoing(self.graph['2_directed'])]
	# 	# print("P outgoing :  ", result_out)
	# 	#
	# 	# self.assertSequenceEqual(result_in, test_in)
	# 	pass
	#
	# def test_positive_centrality_out_2(self):
	# 	# test_out = [1.000, 0.996, 0.996, 0.996, 0.996]  # Not the right values ...
	# 	#
	# 	# result = [round(x, 1) for x in degree_centrality.PositiveCentrality.undirected(self.graph['2_undirected'])]
	# 	# print("P undirected :", result)
	# 	# result_in = [round(x, 1) for x in degree_centrality.PositiveCentrality.incoming(self.graph['2_directed'])]
	# 	# print("P incoming :  ", result_in)
	# 	# result_out = [round(x, 1) for x in degree_centrality.PositiveCentrality.outgoing(self.graph['2_directed'])]
	# 	# print("P outgoing :  ", result_out)
	# 	#
	# 	# self.assertSequenceEqual(result_out, test_out)
	# 	pass

	def test_negative_centrality_in_2(self):  # Works correctly
		test_in = [0.000, 1.000, 1.000, 1.000, 1.000]
		print("test in :   ", test_in)

		result_in = [round(x, 1) for x in degree_centrality.NegativeCentrality.incoming(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))]
		print("incoming :  ", result_in)
		print()

		self.assertSequenceEqual(result_in, test_in)

	def test_negative_centrality_out_2(self):
		test_out = [1.000, 0.996, 0.996, 0.996, 0.996]
		print("test out :  ", test_out)

		# # g2out = matrix_to_graph(tril(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))
		# # g2out = matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']]))
		# # print("g2out : ", g2out)
		# result_out = [round(x, 1) for x in degree_centrality.NegativeCentrality.outgoing(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))]
		h_star = degree_centrality.NegativeCentrality.outgoing(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))
		result_out = [round(x, 1) for x in h_star]
		print("outgoing :  ", [x for x in h_star])
		print("outgoing :  ", result_out)
		print()

		print(self.graph['2_directed'])

		result = [round(x, 1) for x in degree_centrality.NegativeCentrality.undirected(self.graph['2_undirected'])]
		print("undirected :", result)
		# g2in = matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['2_undirected']])))
		# print("g2in : ", g2in)
		# result_in = [round(x, 1) for x in degree_centrality.NegativeCentrality.incoming(g2in)]
		result_in = [round(x, 1) for x in degree_centrality.NegativeCentrality.incoming(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_undirected']])))]
		print("incoming :  ", result_in)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_5(self):  # Works Correctly

		test = [0.901, 0.861, 0.908, 0.861, 0.841, 0.850, 0.862, 0.902, 0.851, 0.907]
		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['5_undirected'])]

		print(test)
		print(result)

		self.assertSequenceEqual(result, test)

	# def test_PN_centrality_in_5(self):
	# 	"""
	# 	Test data were computed with signnet.
	# 	"""
	#
	# 	print()
	# 	print(self.array['5_directed'])
	# 	print(self.array['5_undirected'])
	#
	# 	test = [1.132926, 1.260525, 1.144659, 1.260525, 1.179987, 1.227421, 1.256844, 1.131042, 1.219903]
	# 	result = [round(x, 6) for x in degree_centrality.PNCentrality.undirected(self.graph['5_undirected'])]
	# 	result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['5_directed']))]
	# 	result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['5_directed']))]
	# 	# result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['5_undirected']]))))]
	# 	# result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(tril(array([[max(col, 0) for col in row] for row in self.array['5_undirected']]))))]
	#
	# 	print("test in :   ", test)
	# 	print("result in : ", result_in)
	# 	print()
	# 	print("undirected :", result)
	# 	print("result out :", result_out)
	#
	# 	self.assertSequenceEqual(result_in, test)

	# def test_PN_centrality_out_5(self):
	# 	"""
	# 	Test data were computed with signnet.
	# 	"""
	#
	# 	test = [1.132926, 1.260525, 1.144659, 1.260525, 1.179987, 1.227421, 1.256844, 1.131042, 1.219903]
	# 	result = [round(x, 6) for x in degree_centrality.PNCentrality.undirected(self.graph['5_undirected'])]
	# 	result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['5_directed']))]
	# 	result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['5_directed']))]
	# 	# result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['5_undirected']]))))]
	# 	# result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(tril(array([[max(col, 0) for col in row] for row in self.array['5_undirected']]))))]
	#
	# 	print("test out :  ", test)
	# 	print("result out :", result_out)
	# 	print()
	# 	print("undirected :", result)
	# 	print("result in : ", result_in)
	#
	# 	self.assertSequenceEqual(result_out, test)

	def test_PN_centrality_in_sampson(self):

		test_in = [0.899, 1.286, 1.156, 0.739, 0.675, 0.836, 0.744, 1.025, 1.119, 0.876, 0.799, 1.096, 0.921, 0.827,
			0.648, 0.333, 0.402, 0.397]
		test_out = [1.122, 0.868, 0.793, 0.625, 0.880, 0.621, 0.598, 0.884, 0.779, 0.840, 0.607, 0.681, 0.733, 0.649,
			0.682, 0.810, 0.940, 0.801]
		test_undirected = [1.025, 1.162, 1.105, 0.944, 0.761, 0.852, 0.809, 0.999, 0.982, 1.071, 0.903, 0.955, 0.883,
			0.873, 0.665, 0.420, 0.483, 0.447]

		print("test out :       ", test_out)
		print("test undirected :", test_undirected)
		print()
		print("test in :        ", test_in)

		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['sampson'])]
		result_in = [round(x, 3) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['sampson']))]
		result_out = [round(x, 3) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['sampson']))]
		# result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]
		# result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(tril(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]

		print("result in :      ", result_in)
		print()
		print("undirected :     ", result)
		print("result out :     ", result_out)

		self.assertSequenceEqual(result_in, test_in)

	def test_PN_centrality_out_sampson(self):

		test_in = [0.899, 1.286, 1.156, 0.739, 0.675, 0.836, 0.744, 1.025, 1.119, 0.876, 0.799, 1.096, 0.921, 0.827, 0.648, 0.333, 0.402, 0.397]
		test_out = [1.122, 0.868, 0.793, 0.625, 0.880, 0.621, 0.598, 0.884, 0.779, 0.840, 0.607, 0.681, 0.733, 0.649, 0.682, 0.810, 0.940, 0.801]
		test_undirected = [1.025, 1.162, 1.105, 0.944, 0.761, 0.852, 0.809, 0.999, 0.982, 1.071, 0.903, 0.955, 0.883, 0.873, 0.665, 0.420, 0.483, 0.447]

		print("test in :        ", test_in)
		print("test undirected :", test_undirected)
		print()
		print("test out :       ", test_out)

		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['sampson'])]
		result_in = [round(x, 3) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['sampson']))]
		result_out = [round(x, 3) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['sampson']))]
		# result_in = [round(x, 6) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]
		# result_out = [round(x, 6) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(tril(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]

		print("result out :     ", result_out)
		print()
		print("undirected :     ", result)
		print("result in :      ", result_in)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_sampson(self):

		result = [round(x, 3) for x in degree_centrality.PNCentrality.undirected(self.graph['sampson'])]

		test_in = [0.899, 1.286, 1.156, 0.739, 0.675, 0.836, 0.744, 1.025, 1.119, 0.876, 0.799, 1.096, 0.921, 0.827, 0.648, 0.333, 0.402, 0.397]
		test_out = [1.122, 0.868, 0.793, 0.625, 0.880, 0.621, 0.598, 0.884, 0.779, 0.840, 0.607, 0.681, 0.733, 0.649, 0.682, 0.810, 0.940, 0.801]
		test_undirected = [1.025, 1.162, 1.105, 0.944, 0.761, 0.852, 0.809, 0.999, 0.982, 1.071, 0.903, 0.955, 0.883, 0.873, 0.665, 0.420, 0.483, 0.447]

		print("test in :        ", test_in)
		print("test out :       ", test_out)
		print()
		print("test undirected :", test_undirected)

		print("undirected :     ", result)
		result_in = [round(x, 1) for x in degree_centrality.PNCentrality.incoming(self.graph['sampson'])]
		# result_in = [round(x, 1) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]

		print()
		print("incoming :       ", result_in)
		result_out = [round(x, 1) for x in degree_centrality.PNCentrality.outgoing(self.graph['sampson'])]
		# result_out = [round(x, 1) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(triu(array([[max(col, 0) for col in row] for row in self.array['sampson']]))))]
		print("outgoing :       ", result_out)

		print()

		self.assertSequenceEqual(result, test_undirected)


if __name__ == '__main__':
	unittest.main()
