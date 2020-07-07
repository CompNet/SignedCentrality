#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains unit tests for the modules of the packages signedcentrality and clustering.

.. seealso: signedcentrality
.. seealso: clustering
"""
from os import stat
from os.path import dirname, exists, basename

from numpy import array
from csv import reader, Sniffer, unix_dialect
from glob import glob
from subprocess import call


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
				names.append(basename(path))

	results = {}
	for i in range(len(files_contents)):
		results[names[i]] = files_contents[i].transpose().flatten()

	print(results)  # For tests.

	return None
