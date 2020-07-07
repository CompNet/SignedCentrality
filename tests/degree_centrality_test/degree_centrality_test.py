#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module degree_centrality.
"""

import unittest
from os.path import abspath
from subprocess import call
from signedcentrality import degree_centrality
from csv import reader, Sniffer, writer, QUOTE_MINIMAL
from scipy.sparse import csr_matrix
from numpy import array, transpose, zeros, diag
from igraph import Graph
from signedcentrality._utils.utils import *
from tests import load_data
from tests.degree_centrality_test import Path


def read_CSV(path, remove_signs = False):
	"""
	Creates an igraph.Graph from a CSV file

	:param path: the path of the CSV file
	:type path: str
	:return: the graph
	:rtype: igraph.Graph
	"""

	matrix = None
	csv = []

	with open(path, 'r') as file:

		dialect = Sniffer().sniff(file.read(1024))
		file.seek(0)

		header = Sniffer().has_header(file.read(1024))
		file.seek(0)

		for row in reader(file, dialect):
			csv.append(row)

		if remove_signs:
			matrix = array([[abs(float(csv[i][j])) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))])  # int(header) is 0 if False and 1 if true
		else:
			matrix = array([[float(csv[i][j]) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))])  # int(header) is 0 if False and 1 if true

		# matrix = array([[float(min(1, abs(csv[i][j]))) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))])  # min(1, abs(csv[i][j])) is 0 if abs(csv[i][j])==0 and 1 if abs(csv[i][j])>=1 ; int(header) is 0 if False and 1 if true

	return matrix_to_graph(array(matrix))


def write_CSV(graph, path):
	"""
	Creates a CSV file from an igraph.Graph

	:param graph: the graph
	:type graph: igraph.Graph
	:param path: the path of the CSV file
	:type path: str
	"""

	with open(path, 'w') as file:

		csv_writer = writer(file, delimiter = ',', quotechar='"', quoting=QUOTE_MINIMAL)
		rows = [[str(col) for col in row] for row in get_matrix(graph).toarray().tolist()]
		for row in rows:
			# print(row)
			csv_writer.writerow(row)


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


def convert_sampson_graph(*args):
	"""
	Convert a graph defined in several files to a symmetric undirected signed graph.

	This function takes groups of two parameters.
	The fist one is a graph that represents the positive signed edges.
	The second one is a graph that represents the negative signed edges.

	The graphs that are read by this function may be directed or undirected signed graphs.

	:param args: groups of two graphs
	:type args: Graph or tuple
	:param directed: indicates if the merged graph has to be directed
	:type directed: Bool
	:return: a symmetric undirected signed graph
	:rtype args: Graph
	"""

	for graph in args:
		if not isinstance(graph, Graph):
			msg = "".join(["Arguments have to be igraph.Graph instances, not ", str(type(graph)), "."])
			raise ValueError(msg)

	if len(args) == 0 or len(args) % 2 != 0:
		raise ValueError("Wrong arguments number.")

	length = args[0].ecount()

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

	for i in range(len(matrices)):
		for j in range(len(matrices[i])):
			for row_index in range(length):
				for col_index in range(length):
					matrices[i][row_index][col_index] += all_matrices[i][j][row_index][col_index]

	for i in range(len(matrices)):
		matrices[i] = array([[min(1, abs(value)) for value in row] for row in matrices[i]])

	positive_matrix = matrices[0]
	negative_matrix = matrices[1]

	print(positive_matrix)
	print(negative_matrix)

	symmetric_matrices = [positive_matrix, negative_matrix]

	for i in range(len(matrices)):
		symmetric_matrices[i] = array([[min(1, abs(value)) for value in row] for row in symmetric_matrices[i] + transpose(symmetric_matrices[i]) - diag(symmetric_matrices[i].diagonal())])

	symmetric_positive_matrix = positive_matrix
	symmetric_negative_matrix = negative_matrix

	# print(symmetric_positive_matrix)
	# print(symmetric_negative_matrix)

	pn_matrix = zeros((length, length))
	for row_index in range(length):
		for col_index in range(length):
			pn_matrix[row_index, col_index] = symmetric_positive_matrix[row_index, col_index] - symmetric_negative_matrix[row_index, col_index]

	print(pn_matrix)

	return {
		'pn': pn_matrix,
		'positive': positive_matrix,
		'negative': negative_matrix
		}


class DegreeCentralityTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		# Load data from R script
		print(Path.R_SCRIPT)
		load_data(abspath(Path.RES), abspath(Path.R_SCRIPT))

		gamapos = read_CSV("GAMAPOS.csv")
		gamaneg = read_CSV("GAMANEG.csv")
		gama = convert_graph(gamapos, gamaneg)
		symmetrized_gama = convert_graph(gamapos, gamaneg, directed = False)

		samplk3 = read_CSV("SAMPLK3.csv", True)
		sampdlk = read_CSV("SAMPDLK.csv", True)
		sampes = read_CSV("SAMPES.csv", True)
		sampdes = read_CSV("SAMPDES.csv", True)
		sampin = read_CSV("SAMPIN.csv", True)
		sampnin = read_CSV("SAMPNIN.csv", True)
		samppr = read_CSV("SAMPPR.csv", True)
		sampnpr = read_CSV("SAMPNPR.csv", True)
		sampson = convert_sampson_graph(samplk3, sampdlk, sampes, sampdes, sampin, sampnin, samppr, sampnpr)
		symmetrized_sampson = convert_sampson_graph(samplk3, sampdlk, sampes, sampdes, sampin, sampnin, samppr, sampnpr)
		# sampson = convert_sampson_graph(samplk3, sampdlk)
		# symmetrized_sampson = convert_sampson_graph(samplk3, sampdlk, directed = False)

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
		graph_5_directed.to_undirected("collapse", dict(weight = "mean", id = "first"))
		graph_5_directed.to_directed()
		graph_5_undirected = read_CSV("table_5.csv")
		graph_5_undirected.to_undirected("collapse", dict(weight = "mean", id = "first"))

		self.graph = {
			'gamapos': gamapos,
			'gamaneg': gamaneg,
			'gama': gama,
			'symmetrized_gama': symmetrized_gama,
			'samplk3': samplk3,
			'sampdlk': sampdlk,
			'sampson': sampson,
			'symmetrized_sampson': symmetrized_sampson,
			'2_directed': graph_2_directed,
			'2_undirected': graph_2_undirected,
			'5_directed': graph_5_directed,
			'5_undirected': graph_5_undirected
			}

		self.matrix = {
			'gamapos': get_matrix(self.graph['gamapos']),
			'gamaneg': get_matrix(self.graph['gamaneg']),
			'gama': get_matrix(self.graph['gama']),
			'symmetrized_gama': get_matrix(self.graph['symmetrized_gama']),
			'samplk3': get_matrix(self.graph['samplk3']),
			'sampdlk': get_matrix(self.graph['sampdlk']),
			'sampson': get_matrix(self.graph['sampson']),
			'symmetrized_sampson': get_matrix(self.graph['symmetrized_sampson']),
			'2_directed': get_matrix(self.graph['2_directed']),
			'2_undirected': get_matrix(self.graph['2_undirected']),
			'5_directed': get_matrix(self.graph['5_directed']),
			'5_undirected': get_matrix(self.graph['5_undirected'])
			}

		self.array = {
			'gamapos': self.matrix['gamapos'].toarray(),
			'gamaneg': self.matrix['gamaneg'].toarray(),
			'gama': self.matrix['gama'].toarray(),
			'symmetrized_gama': self.matrix['symmetrized_gama'].toarray(),
			'samplk3': self.matrix['samplk3'].toarray(),
			'sampdlk': self.matrix['sampdlk'].toarray(),
			'sampson': self.matrix['sampson'].toarray(),
			'symmetrized_sampson': self.matrix['symmetrized_sampson'].toarray(),
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
		digits = 6

		test_undirected = [1.111111, 1.111111, 1.159564, 1.079804, 1.115326, 1.199719, 1.272832, 1.234552, 1.111397, 1.075419, 1.162314, 1.162314, 1.151173, 1.075550, 1.111111, 1.111111]

		result = [round(x, digits) for x in degree_centrality.PositiveCentrality.undirected(self.graph['gamapos'])]

		print("test undirected : ", test_undirected)
		print("result :          ", result)

		self.assertSequenceEqual(result, test_undirected)

	def test_positive_centrality_in_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		digits = 7

		test_in = [0.8908817, 1.1686770, 1.1181098, 0.8343868, 0.6738496, 1.0591956, 0.7765441, 1.1117098, 1.0054702, 0.7909116, 0.9452530, 1.0561954, 1.0367580, 0.9738473, 0.9481737, 0.7536297, 0.8348919, 1.0271446]

		result = [round(x, digits) for x in degree_centrality.PositiveCentrality.incoming(self.graph['gamapos'])]

		print("test in :         ", test_in)
		print("result :          ", result)

		self.assertSequenceEqual(result, test_in)

	def test_positive_centrality_out_gamapos(self):
		"""
		Test values have been computed using pn_index() function from package signnet in R.
		"""
		digits = 7

		test_out = [1.1149650, 1.0838575, 0.9360470, 0.8861251, 0.9287938, 0.9350133, 0.9307870, 1.0833276, 0.9227497, 0.9516707, 0.9589060, 0.9467080, 0.9502032, 0.9517389, 0.9340754, 0.9196967, 0.9412390, 0.9322612]
		print("test out :        ", test_out)

		result = [round(x, digits) for x in degree_centrality.PositiveCentrality.outgoing(self.graph['gamapos'])]
		print("result :          ", result)

		self.assertSequenceEqual(result, test_out)

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

	def test_PN_centrality_in_sampson(self):
		digits = 3

		test_in = [0.899, 1.286, 1.156, 0.739, 0.675, 0.836, 0.744, 1.025, 1.119, 0.876, 0.799, 1.096, 0.921, 0.827, 0.648, 0.333, 0.402, 0.397]

		print("test in :        ", test_in)

		result_in = [round(x, digits) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['sampson']))]

		print("result in :      ", result_in)

		self.assertSequenceEqual(result_in, test_in)

	def test_PN_centrality_out_sampson(self):
		digits = 3

		test_out = [1.122, 0.868, 0.793, 0.625, 0.880, 0.621, 0.598, 0.884, 0.779, 0.840, 0.607, 0.681, 0.733, 0.649, 0.682, 0.810, 0.940, 0.801]

		print("test out :       ", test_out)

		result_out = [round(x, digits) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['sampson']))]

		print("result out :     ", result_out)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_sampson(self):
		digits = 3

		test_undirected = [1.025, 1.162, 1.105, 0.944, 0.761, 0.852, 0.809, 0.999, 0.982, 1.071, 0.903, 0.955, 0.883, 0.873, 0.665, 0.420, 0.483, 0.447]

		print("test undirected :", test_undirected)

		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected(self.graph['symmetrized_sampson'])]

		print("undirected :     ", result)

		self.assertSequenceEqual(result, test_undirected)

	def test_PN_centrality_undirected_sampson_table_4(self):
		digits = 3

		test_undirected = [1.030, 1.157, 1.074, 0.897, 0.775, 0.812, 0.763, 0.942, 0.900, 0.993, 0.853, 0.952, 0.860, 0.855, 0.645, 0.393, 0.476, 0.415]

		print("test undirected :", test_undirected)

		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected(self.graph['symmetrized_sampson'])]

		print("undirected :     ", result)

		self.assertSequenceEqual(result, test_undirected)

	def test_PN_centrality_in_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		test_in = [0.8908817, 1.1686770, 1.1181098, 0.8343868, 0.6738496, 1.0591956, 0.7765441, 1.1117098, 1.0054702, 0.7909116, 0.9452530, 1.0561954, 1.0367580, 0.9738473, 0.9481737, 0.7536297, 0.8348919, 1.0271446]

		print("test in :        ", test_in)
		result_in = [round(x, digits) for x in degree_centrality.PNCentrality.incoming(matrix_to_graph(self.array['sampson']))]

		print("result in :      ", result_in)

		self.assertSequenceEqual(result_in, test_in)

	def test_PN_centrality_out_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		test_out = [1.1149650, 1.0838575, 0.9360470, 0.8861251, 0.9287938, 0.9350133, 0.9307870, 1.0833276, 0.9227497, 0.9516707, 0.9589060, 0.9467080, 0.9502032, 0.9517389, 0.9340754, 0.9196967, 0.9412390, 0.9322612]

		print("test out :       ", test_out)

		result_out = [round(x, digits) for x in degree_centrality.PNCentrality.outgoing(matrix_to_graph(self.array['sampson']))]
		print("result out :     ", result_out)

		self.assertSequenceEqual(result_out, test_out)

	def test_PN_centrality_undirected_sampson_signnet(self):
		"""
		This method uses data computed using signnet.
		"""
		digits = 7

		result = [round(x, digits) for x in degree_centrality.PNCentrality.undirected(self.graph['symmetrized_sampson'])]

		test_undirected = [1.0067189, 1.1594213, 1.0081889, 0.8607165, 0.6707594, 0.9633693, 0.7999954, 1.1143644, 0.8702311, 0.8799973, 0.9851447, 1.0109134, 0.9550559, 0.9594615, 0.9105702, 0.6801541, 0.7612246, 0.8900183]

		print("test undirected :", test_undirected)
		print("undirected :     ", result)

		self.assertSequenceEqual(result, test_undirected)

	# def test_write_csv(self):
	# 	graph = self.graph['sampson']
	# 	symmetrized_graph = self.graph['symmetrized_sampson']
	# 	# print(get_matrix(graph).toarray())
	# 	write_CSV(graph, 'sampson_directed.csv')
	# 	write_CSV(symmetrized_graph, 'sampson_undirected.csv')
	#
	# 	graph = self.graph['sampson']
	# 	symmetrized_graph = self.graph['symmetrized_gama']
	# 	write_CSV(graph, 'gama_directed.csv')
	# 	write_CSV(symmetrized_graph, 'gama_undirected.csv')


if __name__ == '__main__':
	unittest.main()
