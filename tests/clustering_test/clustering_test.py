#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for tests.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

import unittest
from os import getcwd, chdir
from sklearn.svm import SVC, SVR
from signedcentrality.clustering import SVCKernel, ClassifierMode, ClassifierData
from signedcentrality.clustering.classifier import Classifier
from signedcentrality.clustering.classifier_comparison import ClassifierComparator
from tests.clustering_test import Path

global_comparator = None
"""
Datasets to train the classifier.

This variable is global to compute it only one time.
"""


def init_svm():
	"""
	Initialize dataset to train the classifier

	:return: the data
	"""

	global global_comparator

	if global_comparator is not None:
		return global_comparator

	# Change the path of unit tests working directory:
	chdir("/".join([getcwd(), Path.TESTS_RES_PATH]))
	Path.load()

	global_comparator = ClassifierComparator()

	return global_comparator


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.comparator = init_svm()

	def test_comparator(self):
		self.comparator.compare_classifiers()

	def test_optimal_classifier(self):
		mode = ClassifierMode.SINGLE_CLASS
		train_data = self.comparator.train[mode][ClassifierData.INPUT]
		train_target = self.comparator.train[mode][ClassifierData.TARGET]
		validation_data = self.comparator.validation[mode][ClassifierData.INPUT]
		validation_target = self.comparator.validation[mode][ClassifierData.TARGET]

		classifier = Classifier(SVC(kernel=SVCKernel.LINEAR), mode, train_data, train_target, validation_data, validation_target)
		classifier.train()
		self.comparator.classifier_test(classifier)


if __name__ == '__main__':
	unittest.main()
