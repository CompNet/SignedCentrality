#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for tests.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

import unittest
from os import walk
from os.path import dirname, splitext, basename
from subprocess import call
from sklearn.svm import SVC
from clustering import SVCKernel
from clustering.classifier import Classifier
from signedcentrality import eigenvector_centrality, degree_centrality
from signedcentrality._utils.utils import *
from tests.clustering_test import Path
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL


def load_data(path: str = None):
	"""
	Load dataset to train and test a Classifier.

	:param path: Path to the dataset
	:type path: str
	:return: the loaded and parsed data
	"""

	# Create a file containing paths to the graphs of the dataset:
	files_paths = []
	for (dir_path, dir_names, file_names) in walk(Path.INPUTS_PATH):
		for file_name in file_names:
			if splitext(basename(file_name))[1] == Path.DEFAULT_EXT:
				file_path = dir_path + '/' + file_name
				files_paths.append(file_path)

	with open(Path.GENERATED_CSV_PATHS_FILE, 'w') as file:
		csv_writer = writer(file, delimiter = ',', quotechar = '"', quoting = QUOTE_MINIMAL)
		csv_writer.writerows([[file_path] for file_path in files_paths])

	# Compute the descriptors :
	call([Path.R_SCRIPT, dirname(Path.R_SCRIPT), Path.RES_PATH, Path.GENERATED_RES_PATH, Path.R_GENERATED_RES_PATH, path, Path.GENERATED_CSV_PATHS_FILE])

	# TODO : load descriptors computed by the R script.


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data = load_data(Path.DATASET_SAMPLE_PATH)

	def test_classifier(self):
		"""
		Test classifiers.

		Each classifier has been set with different parameters.

		:return: a list of results for all classifiers
		"""
		# classifiers = [
		# 	SVC(kernel = SVCKernel.LINEAR),
		# 	SVC(kernel = SVCKernel.POLY),
		# 	SVC(kernel = SVCKernel.SIGMOID),
		# 	SVC(kernel = SVCKernel.PRECOMPUTED),
		# 	SVC()
		# 	]
		# test_results = []
		#
		# for c in classifiers:
		# 	classifier = Classifier(c, self.data)
		# 	results = classifier.train()
		# 	test_results.append(results)
		#
		# 	for key, result in results:
		# 		print(key, " :\t", result, end = "\n\n")

	pass


if __name__ == '__main__':
	unittest.main()
