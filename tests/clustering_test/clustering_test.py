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


main_data = None
"""
Datasets to train the classifier.

This variable is global to compute it only one time.
"""


def init_svm():
	"""
	Initilaize dataset to train the classifier

	:return: the data
	"""

	global main_data

	if main_data is not None:
		return main_data

	# Change the path of unit tests working directory:
	# print(getcwd())
	chdir("/".join([getcwd(), Path.TESTS_RES_PATH]))
	# print(getcwd())
	Path.load()
	# print('res :', Path.RES_PATH)

	training_data, target_data = load_data(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.DEFAULT_SAMPLE_RESULTS_PATH)

	# Tests :
	print("---- Training Data ----")
	print()
	for input_graph, xml in training_data.items():
		print("=>", input_graph)
		for key, value in xml.items():
			print(key, ": ", value, sep="")
		print()
		print()

	print("---- Target Data ----")
	print()
	if target_data is not  None:
		for input_graph, values in target_data.items():
			print("=>", input_graph)
			for value in values:
				print(value, sep="")
			print()
			print()
	else:
		print('No target data.')

	main_data = (training_data, target_data)

	return main_data


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data = init_svm()

	def test_classifier_default_kernel(self):

		classifier = Classifier(SVC(), *self.data)
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_linear_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.LINEAR), *self.data)
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_poly_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.POLY), *self.data)
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_sigmoid_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.SIGMOID), *self.data)
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_pre_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.PRECOMPUTED), *self.data)
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")


if __name__ == '__main__':
	unittest.main()
