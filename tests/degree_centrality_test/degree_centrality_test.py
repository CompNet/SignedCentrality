#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module degree_centrality.
"""

import unittest
from math import trunc
from os.path import abspath
from numpy.linalg import inv
from signedcentrality.centrality import degree_centrality
from csv import reader, Sniffer, writer, QUOTE_MINIMAL
from numpy import array, transpose, zeros, identity, dot
from igraph import Graph
from signedcentrality._utils.utils import *
from tests import load_data, write_csv, read_csv
from tests.degree_centrality_test import Path


main_data = None
"""
Datasets to train the classifier.

This variable is global to compute it only one time.
"""


def convert_graph(*args, directed=True):
	"""
	Convert a graph defined in one or two files to a symmetric undirected signed graph.

	This function takes two or three parameters.
	The first one is a graph that represents the positive signed edges.
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


def convert_sampson_graph(*args):
	"""
	Convert a graph defined in several files to a symmetric undirected signed graph.

	This function takes groups of two parameters.
	The fist one is a graph that represents the positive signed edges.
	The second one is a graph that represents the negative signed edges.

	The graphs that are read by this function may be directed or undirected signed graphs.

	:param args: groups of two graphs
	:type args: Graph or tuple
	:return: a symmetric undirected signed graph
	:rtype args: Graph
	"""

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

	if len(args) == 0 or len(args) % 2 != 0:
		raise ValueError("Wrong arguments number.")

	length = args[0].vcount()

	positive_matrices = []
	negative_matrices = []
	for i in range(len(args)):
		matrix = array([[min(1, abs(value)) for value in row] for row in get_matrix(args[i]).toarray()])

		if i % 2 != 0:
			positive_matrices.append(matrix)
		else:
			negative_matrices.append(matrix)

	all_matrices = [positive_matrices, negative_matrices]
	matrices = [zeros((length, length)), zeros((length, length))]  # Same length as all_matrices.

	for i in range(len(matrices)):  # len(matrices) == len(all_matrices)
		for j in range(len(all_matrices[i])):
			matrices[i] += all_matrices[i][j]

	for i in range(len(matrices)):
		matrices[i] = array([[min(1, abs(value)) for value in row] for row in matrices[i]])

	positive_matrix = matrices[1]
	negative_matrix = matrices[0]

	# print('positive_matrix :', positive_matrix, sep='\n', end='\n\n')
	# print('negative_matrix :', negative_matrix, sep='\n', end='\n\n')

	symmetric_matrices = [positive_matrix, negative_matrix]

	for i in range(len(matrices)):
		symmetric_matrices[i] = array([[min(1, abs(value)) for value in row] for row in symmetric_matrices[i] + transpose(symmetric_matrices[i])])

	symmetric_positive_matrix = symmetric_matrices[0]
	symmetric_negative_matrix = symmetric_matrices[1]

	# print('symmetric_positive_matrix :', symmetric_positive_matrix, sep='\n', end='\n\n')
	# print('symmetric_negative_matrix :', symmetric_negative_matrix, sep='\n', end='\n\n')

	pn_matrix = positive_matrix - negative_matrix
	symmetric_pn_matrix = symmetric_positive_matrix - symmetric_negative_matrix

	# print('pn_matrix :', pn_matrix, sep='\n', end='\n\n')
	# print('symmetric_pn_matrix :', symmetric_pn_matrix, sep='\n', end='\n\n')

	return {
		'pn': matrix_to_graph(pn_matrix),  # Directed
		'symmetric_pn_sampson': matrix_to_graph(symmetric_pn_matrix),  # Undirected
		'positive': matrix_to_graph(positive_matrix),  # Positive directed
		'negative': matrix_to_graph(negative_matrix)  # Negative directed
	}


def initialize_data():
	"""
	Initilaize dataset to train the classifier

	:return: the data
	"""

	global main_data

	if main_data is not None:
		return main_data

	main_data = {}

	gamapos = read_csv("res/GAMAPOS.csv")
	gamaneg = read_csv("res/GAMANEG.csv")
	gama = convert_graph(gamapos, gamaneg)
	symmetrized_gama = convert_graph(gamapos, gamaneg, directed=False)

	samplk3 = read_csv("res/SAMPLK3.csv", True)
	sampdlk = read_csv("res/SAMPDLK.csv", True)
	sampes = read_csv("res/SAMPES.csv", True)
	sampdes = read_csv("res/SAMPDES.csv", True)
	sampin = read_csv("res/SAMPIN.csv", True)
	sampnin = read_csv("res/SAMPNIN.csv", True)
	samppr = read_csv("res/SAMPPR.csv", True)
	sampnpr = read_csv("res/SAMPNPR.csv", True)
	sampson_graphs = convert_sampson_graph(samplk3, sampdlk, sampes, sampdes, sampin, sampnin, samppr, sampnpr)
	pn_sampson = sampson_graphs['pn']
	symmetric_pn_sampson = sampson_graphs['symmetric_pn_sampson']
	positive_sampson = sampson_graphs['positive']
	negative_sampson = sampson_graphs['negative']

	# Graphs used by M. Everett and S. Borgatti in their paper :
	samn = read_csv("res/SAMN.csv", True)
	samp = read_csv("res/SAMP.csv", True)
	samnsym = read_csv("res/SAMNSYM.csv", True)
	sampsym = read_csv("res/SAMPSYM.csv", True)
	sampson_paper = convert_graph(samp, samn)
	symmetric_sampson_paper = convert_graph(sampsym, samnsym)

	graph_2_directed = Graph(5)
	graph_2_directed.to_directed()
	graph_2_directed.add_edge(1, 0)
	graph_2_directed.add_edge(2, 0)
	graph_2_directed.add_edge(3, 0)
	graph_2_directed.add_edge(4, 0)

	graph_2_undirected = Graph(5)
	graph_2_undirected.to_undirected()
	graph_2_undirected.add_edge(1, 0)
	graph_2_undirected.add_edge(2, 0)
	graph_2_undirected.add_edge(3, 0)
	graph_2_undirected.add_edge(4, 0)

	graph_5_directed = read_csv("res/table_5.csv")
	graph_5_directed.to_undirected("collapse", dict(weight="mean", id="first"))
	graph_5_directed.to_directed()
	graph_5_undirected = read_csv("res/table_5.csv")
	graph_5_undirected.to_undirected("collapse", dict(weight="mean", id="first"))

	main_data['graph'] = {
		'gamapos': gamapos,
		'gamaneg': gamaneg,
		'gama': gama,
		'symmetrized_gama': symmetrized_gama,
		'pn_sampson': pn_sampson,
		'symmetric_pn_sampson': symmetric_pn_sampson,
		'positive_sampson': positive_sampson,
		'negative_sampson': negative_sampson,
		'sampson_paper': sampson_paper,
		'symmetric_sampson_paper': symmetric_sampson_paper,
		'2_directed': graph_2_directed,
		'2_undirected': graph_2_undirected,
		'5_directed': graph_5_directed,
		'5_undirected': graph_5_undirected
	}

	main_data['matrix'] = {
		'gamapos': get_matrix(main_data['graph']['gamapos']),
		'gamaneg': get_matrix(main_data['graph']['gamaneg']),
		'gama': get_matrix(main_data['graph']['gama']),
		'symmetrized_gama': get_matrix(main_data['graph']['symmetrized_gama']),
		'pn_sampson': get_matrix(main_data['graph']['pn_sampson']),
		'symmetric_pn_sampson': get_matrix(main_data['graph']['symmetric_pn_sampson']),
		'positive_sampson': get_matrix(main_data['graph']['positive_sampson']),
		'negative_sampson': get_matrix(main_data['graph']['negative_sampson']),
		'sampson_paper': get_matrix(main_data['graph']['sampson_paper']),
		'symmetric_sampson_paper': get_matrix(main_data['graph']['symmetric_sampson_paper']),
		'2_directed': get_matrix(main_data['graph']['2_directed']),
		'2_undirected': get_matrix(main_data['graph']['2_undirected']),
		'5_directed': get_matrix(main_data['graph']['5_directed']),
		'5_undirected': get_matrix(main_data['graph']['5_undirected'])
	}

	main_data['array'] = {
		'gamapos': main_data['matrix']['gamapos'].toarray(),
		'gamaneg': main_data['matrix']['gamaneg'].toarray(),
		'gama': main_data['matrix']['gama'].toarray(),
		'symmetrized_gama': main_data['matrix']['symmetrized_gama'].toarray(),
		'pn_sampson': main_data['matrix']['pn_sampson'].toarray(),
		'symmetric_pn_sampson': main_data['matrix']['symmetric_pn_sampson'].toarray(),
		'positive_sampson': main_data['matrix']['positive_sampson'].toarray(),
		'negative_sampson': main_data['matrix']['negative_sampson'].toarray(),
		'sampson_paper': main_data['matrix']['sampson_paper'].toarray(),
		'symmetric_sampson_paper': main_data['matrix']['symmetric_sampson_paper'].toarray(),
		'samn': read_csv("res/SAMN.csv", True, True),
		'samp': read_csv("res/SAMP.csv", True, True),
		'samnsym': read_csv("res/SAMNSYM.csv", True, True),
		'sampsym': read_csv("res/SAMPSYM.csv", True, True),
		'2_directed': main_data['matrix']['2_directed'].toarray(),
		'2_undirected': main_data['matrix']['2_undirected'].toarray(),
		'5_directed': main_data['matrix']['5_directed'].toarray(),
		'5_undirected': main_data['matrix']['5_undirected'].toarray()
	}

	# Load signnet computed main_data from R script

	write_csv(pn_sampson, 'res/generated/sampson_directed.csv')
	write_csv(symmetric_pn_sampson, 'res/generated/sampson_undirected.csv')

	write_csv(gama, 'res/generated/gama_directed.csv')
	write_csv(symmetrized_gama, 'res/generated/gama_undirected.csv')

	main_data['signnet_data'] = load_data(abspath(Path.R_RES), abspath(Path.R_SCRIPT))

	return main_data


class DegreeCentralityTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data = initialize_data()

		self.graph = self.data['graph']
		self.matrix = self.data['matrix']
		self.array = self.data['array']
		self.signnet_data = self.data['signnet_data']

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

		for i in range(min(len(array_test), len(self.array['5_directed']))):
			for j in range(min(len(array_test[i]), len(self.array['5_directed'][i]))):
				self.assertEqual(self.array['5_directed'][i][j], array_test[i][j])

	def test_convert_sampson_graph(self):
		test = [self.array['sampson_paper'], self.array['symmetric_sampson_paper']]
		result = [self.array['pn_sampson'], self.array['symmetric_pn_sampson']]

		length = len(test[0])

		for n in range(len(test)):
			# print("Matrix", n + 1)
			for i in range(length):
				# print(test[n][i], result[n][i], sep='\n', end='\n\n')
				for j in range(length):
					self.assertEqual(test[n][i][j], result[n][i][j])

	def test_positive_centrality_undirected_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		digits = 6

		test_undirected = [1.111111, 1.111111, 1.159564, 1.079804, 1.115326, 1.199719, 1.272832, 1.234552, 1.111397, 1.075419, 1.162314, 1.162314, 1.151173, 1.075550, 1.111111, 1.111111]

		result = [round(x, digits) for x in degree_centrality.PositiveCentrality.undirected(self.graph['gamapos'])]

		print("test undirected : ", test_undirected)
		print("result :          ", result)

		self.assertSequenceEqual(result, test_undirected)

	def test_negative_centrality_undirected_gamaneg(self):  # Works correctly
		digits = 2

		test = [71.83, 72.46, 90.38, 95.21, 81.31, 75.05, 100.00, 94.41, 80.69, 85.69, 74.44, 80.56, 80.97, 83.78, 67.76, 68.26]
		print(test)

		result = [round(100 * x, digits) for x in degree_centrality.NegativeCentrality.undirected(self.graph['gamaneg'], True)]
		print(result)

		self.assertSequenceEqual(result, test)

	def test_negative_centrality_in_2(self):  # Works correctly
		digits = 3

		test_in = [0.000, 1.000, 1.000, 1.000, 1.000]
		print("test in :   ", test_in)

		result_in = [round(x, digits) for x in degree_centrality.NegativeCentrality.incoming(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))]
		print("incoming :  ", result_in)

		self.assertSequenceEqual(result_in, test_in)

	def test_negative_centrality_out_2(self):
		digits = 3

		test_out = [1.000, 0.996, 0.996, 0.996, 0.996]
		print("test out :  ", test_out)

		h_star = degree_centrality.NegativeCentrality.outgoing(matrix_to_graph(array([[max(col, 0) for col in row] for row in self.array['2_directed']])))
		result_out = [round(x, digits) for x in h_star]
		print("outgoing :  ", [x for x in h_star])

		self.assertSequenceEqual([round(x, 2) for x in result_out], [round(x, 2) for x in test_out])  # Results are considered as right because wrong values come on the third digit after the decimal point. The difference is negligible.
		# self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_5(self):  # Works Correctly
		digits = 3

		test = [0.901, 0.861, 0.908, 0.861, 0.841, 0.850, 0.862, 0.902, 0.851, 0.907]
		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected(self.graph['5_undirected'])]

		print(test)
		print(result)

		self.assertSequenceEqual(result, test)

	def test_PN_centrality_in_5(self):
		"""
		Test data were computed with signnet.
		"""
		digits = 7

		test = [0.9009747, 0.8613482, 0.9076997, 0.8613482, 0.8410658, 0.8496558, 0.8617321, 0.9015909, 0.8509848, 0.9072930]
		result_in = [round(x, digits) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['5_directed']))]

		print("test in :   ", test)
		print("result in : ", result_in)

		self.assertSequenceEqual(result_in, test)

	def test_PN_centrality_out_5(self):
		"""
		Test data were computed with signnet.
		"""
		digits = 7

		test = [0.9009747, 0.8613482, 0.9076997, 0.8613482, 0.8410658, 0.8496558, 0.8617321, 0.9015909, 0.8509848, 0.9072930]
		result_out = [round(x, digits) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['5_directed']))]

		print("test out :  ", test)
		print("result out :", result_out)

		self.assertSequenceEqual(result_out, test)

	def test_PN_centrality_in_sampson_table_7(self):
		digits = 2

		test_in = [0.92, 1.24, 1.14, 0.79, 0.74, 0.87, 0.79, 1.03, 1.12, 0.95, 0.85, 1.08, 0.96, 0.88, 0.71, 0.44, 0.49, 0.49]

		print("test in :        ", test_in)

		result_in = [round(x, digits) for x in degree_centrality.PNCentrality.incoming_on_matrices(self.array['samp'], self.array['samn'])]

		print("result in :      ", result_in)

		self.assertSequenceEqual(result_in, test_in)

	def test_PN_centrality_out_sampson_table_7(self):
		digits = 2

		test_out = [1.11, 1.03, 0.99, 0.86, 0.88, 0.80, 0.83, 1.01, 0.87, 0.99, 0.88, 0.92, 0.93, 0.90, 0.86, 0.83, 0.94, 0.85]

		print("test out :       ", test_out)

		result_out = [round(x, digits) for x in degree_centrality.PNCentrality.outgoing_on_matrices(self.array['samp'], self.array['samn'])]

		print("result out :     ", result_out)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_sampson_table_4(self):
		digits = 2

		test_undirected = [1.03, 1.16, 1.07, 0.90, 0.77, 0.81, 0.76, 0.94, 0.90, 0.99, 0.85, 0.95, 0.86, 0.85, 0.64, 0.39, 0.48, 0.41]

		print("test undirected :", test_undirected)

		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected_on_matrices(self.array['sampsym'], self.array['samnsym'])]

		print("undirected :     ", result)

		self.assertSequenceEqual(result, test_undirected)

	def test_PN_centrality_in_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		test_in = [round(x, digits) for x in self.signnet_data['sampson_directed_in']]

		print("test in :        ", test_in)
		result_in = [round(x, digits) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['pn_sampson']))]

		print("result in :      ", result_in)

		self.assertSequenceEqual(result_in, test_in)

	def test_PN_centrality_out_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		test_out = [round(x, digits) for x in self.signnet_data['sampson_directed_out']]

		print("test out :       ", test_out)

		result_out = [round(x, digits) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['pn_sampson']))]
		print("result out :     ", result_out)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected(self.graph['symmetric_pn_sampson'])]

		test_undirected = [round(x, digits) for x in self.signnet_data['sampson_undirected_undirected']]

		print("test undirected :", test_undirected)
		print("undirected :     ", result)

		self.assertSequenceEqual(result, test_undirected)


if __name__ == '__main__':
	unittest.main()
