#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains unit tests for the modules of the packages signedcentrality and clustering.

.. seealso: signedcentrality
.. seealso: clustering
"""
from os import stat
from os.path import dirname, exists, basename, splitext
from numpy import array
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL
from glob import glob
from subprocess import call
from signedcentrality._utils.utils import get_matrix, matrix_to_graph


def read_csv(path, remove_signs=False, return_matrix=False):
	"""
	Creates an igraph.Graph from a CSV file

	:param path: the path of the CSV file
	:type path: str
	:param remove_signs: True if the signs must be removed
	:type remove_signs: bool
	:param return_matrix: True if the function must return the matrix instead of the graph
	:type return_matrix: bool
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

	if return_matrix:
		return array(matrix)

	return matrix_to_graph(array(matrix))


def write_csv(graph, path):
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
			csv_writer.writerow(row)


def load_data(res_path, R_script_path):
	"""
	Loads data computed using a R script

	The script clustering_test.R is used in order to compute data.
	It exports data into CSV files which are read by this function.

	:return: data to make tests
	"""

	call([R_script_path, dirname(R_script_path)])

	path_names = [path for path in glob(res_path)]  # Paths of the files which are exported by the script.
	files_contents = []
	names = []

	for path in path_names:
		csv = []

		if exists(path) and stat(path).st_size != 0:
			with open(path, 'r') as file:

				dialect = Sniffer().sniff(file.read(1024))
				file.seek(0)
				header = Sniffer().has_header(file.read(1024))
				file.seek(0)

				for row in reader(file, dialect):
					csv.append(row)

				files_contents.append(array([[float(csv[i][j]) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))]))  # int(header) is 0 if False and 1 if true.
				names.append(splitext(basename(path))[0])

	results = {}
	for i in range(len(files_contents)):
		results[names[i]] = files_contents[i].transpose().flatten()

	# print(results)  # For tests.

	return results
