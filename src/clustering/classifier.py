#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a classifier which uses centralities computed in signedcentrality package.
"""

from os.path import dirname, exists
from subprocess import call
from typing import Any
from xml.etree.ElementTree import parse, Element, ElementTree, SubElement
from numpy import array, mean
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from clustering import XMLKeys
from os import walk, makedirs, getcwd
from os.path import dirname, splitext, basename
from statistics import mean, stdev
from subprocess import call
from sklearn.svm import SVC
from clustering import SVCKernel
from signedcentrality import eigenvector_centrality, degree_centrality, CentralityMeasure
from signedcentrality._utils.utils import *
from signedcentrality.degree_centrality import PNCentrality
from signedcentrality.eigenvector_centrality import compute_eigenvector_centrality, EigenvectorCentrality
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL
from clustering import Path
from signedcentrality._utils.utils import *


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

	return {'mean': centrality_mean, 'stddev': centrality_stddev}


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

	results = {}
	for centrality_class in centrality_classes:
		centralities = _compute_centrality_mean_stddev(graph, centrality_class)

		results = {
			**results,
			**dict(zip(
				['-'.join([centrality_class.__name__, key]) for key in centralities.keys()],
				[value for value in centralities.values()]
				))
			}

	return results


def read_xml(path):
	"""
	Read an XML file

	This file must contain results written by the R script.

	:param path: path of the XML file
	:type path: str
	:return: results
	:rtype: dict
	"""

	results = {}
	xml_tree = parse(path)
	root = xml_tree.getroot()
	for descriptor in root:
		descriptor_type = descriptor.get(XMLKeys.TYPE)
		descriptor_value = descriptor.get(XMLKeys.VALUE)

		results[descriptor_type] = descriptor_value

	return results


def write_xml(xml_path, paths):
	"""
	Write an XML file.

	This file contains paths to input files which must be read by the R script and results files which are created by the R script.

	:param paths: Paths to input and results files
	:type paths: list of list
	:param xml_path: Path to the file this function creates
	:type xml_path: str
	"""

	makedirs(Path.RES_PATH, exist_ok = True)
	makedirs(Path.GENERATED_RES_PATH, exist_ok = True)
	makedirs(Path.R_GENERATED_RES_PATH, exist_ok = True)

	root = Element(XMLKeys.ROOT)
	for row in paths:
		if len(row) < 2:
			raise IndexError("The list must contain tuples of two elements.")

		SubElement(root, XMLKeys.PATH, **{"" + str(XMLKeys.INPUT_FILE): row[0], "" + str(XMLKeys.RESULT_FILE): row[1]})

	tree = ElementTree(root)
	tree.write(xml_path, encoding = 'utf-8', xml_declaration = True)


def load_data(training_data_directory_path: str, target_directory_path: str, input_files_paths_xml_file: str = Path.GENERATED_XML_PATHS_FILE):
	"""
	Load dataset to train and test a Classifier.

	:param input_files_paths_xml_file: Path to the dataset
	:type input_files_paths_xml_file: str
	:return: the loaded and parsed data
	"""

	# Create a file containing paths to the graphs of the dataset:
	input_files_paths = []
	for (dir_path, dir_names, file_names) in walk(training_data_directory_path):
		for file_name in file_names:
			if splitext(basename(file_name))[1] == Path.DEFAULT_EXT:
				file_path = dir_path + '/' + file_name
				input_files_paths.append(file_path)

	file_paths = [
		[
			input_file_path,  # Path to input file.
			(Path.R_GENERATED_RES_PATH + input_file_path.replace(training_data_directory_path, '')).replace(Path.GRAPHML_EXT,
				Path.XML_EXT)  # Path to R generated file containing results for input file.
			] for input_file_path in input_files_paths
		]

	write_xml(input_files_paths_xml_file, file_paths)

	# # Compute the descriptors :
	# print("test :", getcwd())
	# Path.load(getcwd())
	# print('res :', Path.RES_PATH)
	# print('R_SCRIPT :', Path.R_SCRIPT)
	call([
		Path.R_SCRIPT,  # Path to the script to run
		dirname(Path.R_SCRIPT),  # Current working directory of this script
		input_files_paths_xml_file  # Path to the XML file containing the paths to files whose descriptors must be computed, and files to write the computed descriptors.
		])

	# output_files_paths = []
	# for (dir_path, dir_names, file_names) in walk(Path.R_GENERATED_RES_PATH):
	# 	for file_name in file_names:
	# 		if splitext(basename(file_name))[1] == Path.XML_EXT:
	# 			file_path = dir_path + '/' + file_name
	# 			output_files_paths.append(file_path)

	training_data = {}
	for io_paths in file_paths:
		input_file_path = io_paths[0]
		result_file_path = io_paths[1]

		xml_results = read_xml(result_file_path)
		centralities = compute_centralities_mean_stddev(read_graph(input_file_path))

		training_data = {**training_data, input_file_path: {**xml_results, **centralities}}

	return {'training_data': training_data}


class Classifier:
	"""
	This class computes the number of classes of solutions for a clustering.
	"""

	def __init__(self, classifier: SVC, training_vectors = None, target_values = None, dataset = None):
		"""
		Creates a newly allocated Classifier object.

		:param classifier: SVC classifier which must be used as classifier
		:type classifier: SVC
		"""

		self.data = training_vectors
		self.target = target_values

		self.__classifier = classifier
		self.accuracy_score = None
		self.precision_score = None
		self.recall_score = None
		self.accuracy_list = []
		self.precision_list = []
		self.recall_list = []

		self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(self.data, self.target,
			test_size = .3)

	def train(self):
		"""
		Train and test the classifier

		The classifier in trained with 70% of training data and 30% of test data.
		Data set is randomly divided.

		:return: means for all accuracy, precision and recall scores
		"""

		self.__classifier.fit(
			self.train_data,  # Lists of descriptors
			self.test_target  # List of results for each list of descriptors
			)

		for test_data in self.test_data:
			predicted_test_target = self.__classifier.predict(test_data)
			self.accuracy_list.append(accuracy_score(self.test_target, predicted_test_target))
			self.precision_list.append(precision_score(self.test_target, predicted_test_target))
			self.recall_list.append(recall_score(self.test_target, predicted_test_target))

		self.accuracy_score = mean(self.accuracy_list)
		self.precision_score = mean(self.precision_list)
		self.recall_score = mean(self.recall_list)

		return {'accuracy': self.accuracy_score, 'precision': self.precision_score, 'recall': self.recall_score}
