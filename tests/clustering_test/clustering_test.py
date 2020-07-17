#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for tests.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

import unittest
from os import walk, getcwd, chdir
from os.path import dirname, splitext, basename
from statistics import mean, stdev
from subprocess import call
from sklearn.svm import SVC
from clustering import SVCKernel, XMLKeys
from clustering.classifier import Classifier, load_data
from signedcentrality import eigenvector_centrality, degree_centrality, CentralityMeasure
from signedcentrality._utils.utils import *
from signedcentrality.degree_centrality import PNCentrality
from signedcentrality.eigenvector_centrality import compute_eigenvector_centrality, EigenvectorCentrality
from tests.clustering_test import Path
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		# Change the path of unit tests working directory:
		print(getcwd())
		chdir("/".join([getcwd(), Path.TESTS_RES_PATH]))
		print(getcwd())
		Path.load()
		print('res :', Path.RES_PATH)

		self.data = load_data(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.DEFAULT_SAMPLE_RESULTS_PATH)

		# Tests :
		for train_target, data in self.data.items():
			print("----", train_target, "----")
			print()
			for input_graph, xml in data.items():
				print("=>", input_graph)
				for key, value in xml.items():
					print(key, ": ", value, sep = "")
				print()
				print()

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
