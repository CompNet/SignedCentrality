#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for tests.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

import unittest
from collections import OrderedDict
from os import getcwd, chdir
from sklearn.svm import SVC
from signedcentrality.clustering import SVCKernel, XMLKeys, ClassifierMode, ClassifierData
from signedcentrality.clustering.classifier import Classifier, load_data
from signedcentrality._utils.utils import *
from signedcentrality.centrality.degree_centrality import PNCentrality
from tests.clustering_test import Path

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

	# training_data, target_data = load_data(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.DEFAULT_SAMPLE_CLUSTERS_PATH)
	training_data, target_data = load_data(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.DEFAULT_SAMPLE_RESULTS_PATH)

	# # Tests :
	# print("---- Training Data ----")
	# print()
	# for input_graph, xml in training_data.items():
	# 	print("\t", input_graph, sep="")
	# 	for key, value in xml.items():
	# 		print("\t\t", key, ": ", value, sep="")
	# 	print()
	# 	print()
	#
	# print("---- Target Data ----")
	# print()
	# if target_data is not None:
	# 	for mode, targets in target_data.items():
	# 		print('\t', mode, sep='')
	# 		for input_graph, value in targets.items():
	# 			print("\t\t", input_graph, ':', sep='')
	# 			print('\t\t\t=> ', value, sep="")
	# 			print()
	# 		print()
	#
	# else:
	# 	print('No target data.')

	data_tuples = {
		mode: [
			(
				[value for key, value in OrderedDict(training_data[path]).items()],
				target_data[mode][path]
			) for path in OrderedDict(training_data).keys()
		] for mode in [
			ClassifierMode.SINGLE_CLASS,
			ClassifierMode.CLASSES_NUMBER,
			ClassifierMode.SINGLE_SOLUTION,
			ClassifierMode.SOLUTIONS_NUMBER
		]
	}

	data = {
		mode: {
			ClassifierData.INPUT: [
				input_set for input_set, target_set in (data_tuples[mode])
			],
			ClassifierData.TARGET: [
				target_set for input_set, target_set in (data_tuples[mode])
			],
		} for mode in [
			ClassifierMode.SINGLE_CLASS,
			ClassifierMode.CLASSES_NUMBER,
			ClassifierMode.SINGLE_SOLUTION,
			ClassifierMode.SOLUTIONS_NUMBER
		]
	}

	main_data = (data, training_data, target_data)

	return main_data


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data, self.training_data, self.target_data = init_svm()

		# Tests :
		for mode, value in self.data.items():
			print(mode)
			for data, v in value.items():
				print('\t', data, sep='')
				
				if data == ClassifierData.INPUT:
					for descriptors in v:
						for descriptor in descriptors:
							print('\t\t', descriptor, sep='')
						print()
				
				else:
					for target in v:
						print('\t\t', target, sep='')

	def test_load_data(self):
		"""
		Test if the sets have the same keys
		"""

		self.assertSequenceEqual(self.training_data.keys(), self.target_data[ClassifierMode.SINGLE_CLASS].keys())
		self.assertSequenceEqual(self.training_data.keys(), self.target_data[ClassifierMode.CLASSES_NUMBER].keys())
		self.assertSequenceEqual(self.training_data.keys(), self.target_data[ClassifierMode.SINGLE_SOLUTION].keys())
		self.assertSequenceEqual(self.training_data.keys(), self.target_data[ClassifierMode.SOLUTIONS_NUMBER].keys())

	def test_classifier_default_kernel(self):

		classifier = Classifier(SVC(), mode=ClassifierMode.SINGLE_CLASS, *(self.data[ClassifierMode.SINGLE_CLASS]))
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_linear_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.LINEAR), mode=ClassifierMode.SINGLE_CLASS, *(self.data[ClassifierMode.SINGLE_CLASS]))
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_poly_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.POLY), mode=ClassifierMode.SINGLE_CLASS, *(self.data[ClassifierMode.SINGLE_CLASS]))
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_sigmoid_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.SIGMOID), mode=ClassifierMode.SINGLE_CLASS, *(self.data[ClassifierMode.SINGLE_CLASS]))
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	def test_classifier_pre_kernel(self):

		classifier = Classifier(SVC(kernel=SVCKernel.PRECOMPUTED), mode=ClassifierMode.SINGLE_CLASS, *(self.data[ClassifierMode.SINGLE_CLASS]))
		results = classifier.train()

		for key, result in results:
			print(key, " :\t", result, end="\n\n")


if __name__ == '__main__':
	unittest.main()
