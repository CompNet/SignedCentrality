#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a classifier which uses centralities computed in signedcentrality package.
"""

from os.path import dirname
from subprocess import call
from typing import Any
from numpy import array, mean
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from clustering import Path


class Classifier:
	"""

	"""

	def __init__(self, classifier: SVC, dataset = None):
		"""
		Creates a newly allocated Classifier object.

		:param classifier: SVC classifier which must be used as classifier
		:type classifier: SVC
		"""

		self.data = dataset

		self.__classifier = classifier
		self.accuracy_score = None
		self.precision_score = None
		self.recall_score = None
		self.accuracy_list = []
		self.precision_list = []
		self.recall_list = []

		self.train_tests = train_test_split(self.data, test_size = .3)
		self.train_sets = []
		self.test_sets = []
		for i in range(len(self.train_tests)):
			if i % 2 != 0:
				self.train_sets.append(self.train_tests[i])
			else:
				self.test_sets.append(self.train_tests[i])

	def train(self):
		"""
		Train and test the classifier

		The classifier in trained with 70% of training data and 30% of test data.
		Data set is randomly divided.

		:return: means for all accuracy, precision and recall scores
		"""

		self.__classifier.fit(self.train_sets)

		for test in self.test_sets:
			predicted = self.__classifier.predict(test)
			self.accuracy_list.append(accuracy_score(test, predicted))
			self.precision_list.append(precision_score(test, predicted))
			self.recall_list.append(recall_score(test, predicted))

		self.accuracy_score = mean(self.accuracy_list)
		self.precision_score = mean(self.precision_list)
		self.recall_score = mean(self.recall_list)

		return {'accuracy': self.accuracy_score, 'precision': self.precision_score, 'recall': self.recall_score}

