#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.
"""

import unittest
from os.path import abspath, dirname

from signedcentrality import eigenvector_centrality, degree_centrality
from signedcentrality._utils.utils import *
from tests import Path
from subprocess import call
from glob import glob
from csv import reader, Sniffer
from igraph import Graph
from numpy import array


def load_data():
	"""
	Loads data computed using a R script

	The script clustering_test.R is used in order to compute data.
	It exports data into CSV files which are read by this function.

	:return: data to make tests
	"""

	# r_script_directory = abspath(dirname(Path.R_SCRIPT))
	call(Path.R_SCRIPT + ' ' + Path.RES)  # Call R script to export data to use in tests into RES directory.

	path_names = [path for path in glob(Path.RES)]  # Paths of the files which are exported by the script.
	files_contents = []

	for path in path_names:
		csv = []

		with open(path, 'r') as file:

			dialect = Sniffer().sniff(file.read(1024))
			file.seek(0)
			header = Sniffer().has_header(file.read(1024))
			file.seek(0)

			for row in reader(file, dialect):
				csv.append(row)

			files_contents.append(array([[float(csv[i][j]) for j in range(int(header), len(csv[i]))] for i in range(int(header), len(csv))]))  # int(header) is 0 if False and 1 if true

	# TODO: parse data.
	print(files_contents)  # For tests.

	return None


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		# self.data = load_data()
		load_data()

	def stub(self):
		self.assertEqual(True, False)


if __name__ == '__main__':
	unittest.main()
