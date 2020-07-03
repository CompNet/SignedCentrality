#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a classifier which uses centralities computed in signedcentrality package.
"""

from numpy import array, mean
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from clustering import SVCKernel


def test_classifier(data, classifier):
	"""
	Train and test a given classifier with the given data set

	The classifier in trained with 70% of training data and 30% of test data.
	Data set is randomly divided.

	:param data: data set which have to be used to train and test the classifier.
	:param classifier: classifier that must be tested
	:return: means for all accuracy, precision and recall scores
	"""
	train_tests = train_test_split(data, test_size = .3)
	train_sets = []
	test_sets = []
	for i in range(len(train_tests)):
		if i % 2 != 0:
			train_sets.append(train_tests[i])
		else:
			test_sets.append(train_tests[i])

	svm_classifier = classifier
	svm_classifier.fit(train_sets)

	accuracy_list = []
	precision_list = []
	recall_list = []

	for test in test_sets:
		predicted = svm_classifier.predict(test)
		accuracy_list.append(accuracy_score(test, predicted))
		precision_list.append(precision_score(test, predicted))
		recall_list.append(recall_score(test, predicted))

	accuracy = mean(accuracy_list)
	precision = mean(precision_list)
	recall = mean(recall_list)

	return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


def test_classifiers(data):
	"""
	Test classifiers.

	Each classifier has been set with different parameters.

	:param data: data set which have to be used to train and test the classifier.
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

	for classifier in classifiers:
		results = test_classifier(classifier)
		test_results.append(results)

		for key, result in results:
			print(key, " :\t", result, end="\n\n")

	return test_results
