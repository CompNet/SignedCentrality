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
from os import walk
from os.path import dirname, splitext, basename
from statistics import mean, stdev
from subprocess import call
from sklearn.svm import SVC
from clustering import SVCKernel
from clustering.classifier import Classifier
from signedcentrality import eigenvector_centrality, degree_centrality, CentralityMeasure
from signedcentrality._utils.utils import *
from signedcentrality.degree_centrality import PNCentrality
from signedcentrality.eigenvector_centrality import compute_eigenvector_centrality, EigenvectorCentrality
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL


def _compute_centrality_mean_stddev(graph, centrality_class):
	"""
	Compute mean and standard deviation for the given measure of centrality

	It is computed with the undirected() method of the class centrality_class.
	The class centrality_class must be a class which computes a measure of centrality.
	Such a class is a subclass of CentralityMeasure.

	:param graph: graph whose centrality must be computed
	:type graph: igraph.Graph
	:param centrality_class: class that must be used for centrality computing
	:type centrality_class: class
	:return: the mean and the standard deviation
	"""

	if type(centrality_class) != type(CentralityMeasure):
		raise TypeError("The second parameter must be a CentralityMeasure subclass.")

	centrality = centrality_class.undirected(graph)
	centrality_mean = mean(centrality)
	centrality_stddev = stdev(centrality)

	return centrality_mean, centrality_stddev


def compute_centralities_mean_stddev(graph):
	"""
	Compute mean and standard deviation for all measures of centrality

	:param graph: graph whose centrality must be computed
	:type graph: igraph.Graph
	:return: a lis of means and standard deviations of these centralities
	"""

	centrality_classes = [
		EigenvectorCentrality,
		PNCentrality
		]

	return [_compute_centrality_mean_stddev(graph, centrality_class) for centrality_class in centrality_classes]


def load_data(path: str = None):
	"""
	Load dataset to train and test a Classifier.

	:param path: Path to the dataset
	:type path: str
	:return: the loaded and parsed data
	"""

	# Create a file containing paths to the graphs of the dataset:
	input_files_paths = []
	for (dir_path, dir_names, file_names) in walk(Path.INPUTS_PATH):
		for file_name in file_names:
			if splitext(basename(file_name))[1] == Path.DEFAULT_EXT:
				file_path = dir_path + '/' + file_name
				input_files_paths.append(file_path)

	# # output_file_paths = []
	# # for input_file_path in input_files_paths:
	# # 	output_file_path = input_file_path.replace(Path.INPUTS_PATH, '')
	# # 	output_file_paths.append(output_file_path)
	# #
	# output_file_paths = [input_file_path.replace(Path.INPUTS_PATH, '') for input_file_path in input_files_paths]

	with open(Path.GENERATED_CSV_PATHS_FILE, 'w') as file:
		csv_writer = writer(file, delimiter = ',', quotechar = '"', quoting = QUOTE_MINIMAL)
		csv_writer.writerows([
			[
				input_file_path,  # Path to input file.
				Path.R_GENERATED_RES_PATH + input_file_path.replace(Path.INPUTS_PATH, '')  # Path to R generated file containing results for input file.
			] for input_file_path in input_files_paths])

	# Compute the descriptors :
	call([Path.R_SCRIPT, dirname(Path.R_SCRIPT), Path.RES_PATH, Path.GENERATED_RES_PATH, Path.R_GENERATED_RES_PATH, path, Path.GENERATED_CSV_PATHS_FILE])

	# TODO : load descriptors computed by the R script.


class Classifier:
	"""
	This class computes the number of classes of solutions for a clustering.
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

