#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for tests.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

import unittest

from sklearn.svm import SVC

from clustering import SVCKernel
from clustering.classifier import Classifier
from signedcentrality import eigenvector_centrality, degree_centrality
from signedcentrality._utils.utils import *
from tests import load_data
from clustering import Path


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data = load_data(Path.DATASET_PATH)

	def test_classifier(self):
		"""
		Test classifiers.

		Each classifier has been set with different parameters.

		:return: a list of results for all classifiers
		"""
		classifiers = [
			SVC(kernel = SVCKernel.LINEAR),
			SVC(kernel = SVCKernel.POLY),
			SVC(kernel = SVCKernel.SIGMOID),
			SVC(kernel = SVCKernel.PRECOMPUTED),
			SVC()
			]
		test_results = []

		for c in classifiers:
			classifier = Classifier(c, self.data)
			results = classifier.train()
			test_results.append(results)

			for key, result in results:
				print(key, " :\t", result, end = "\n\n")


if __name__ == '__main__':
	unittest.main()
