#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a classifier which uses centralities computed in centrality package.
"""

from os.path import dirname, exists
from subprocess import call
from sys import stderr
from typing import Any
from xml.etree.ElementTree import parse, Element, ElementTree, SubElement
from numpy import array, mean, ndarray
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from signedcentrality.clustering import XMLKeys, CSVResultsFileColNames, ClassifierMode
from os import walk, makedirs, getcwd, system
from os.path import dirname, splitext, basename
from statistics import mean, stdev
from subprocess import call
from sklearn.svm import SVC
from signedcentrality.clustering import SVCKernel
from signedcentrality.centrality import eigenvector_centrality, CentralityMeasure
from signedcentrality.centrality import degree_centrality
from signedcentrality._utils.utils import *
from signedcentrality.centrality.degree_centrality import PNCentrality
from signedcentrality.centrality.eigenvector_centrality import compute_eigenvector_centrality, EigenvectorCentrality
from csv import reader, Sniffer, unix_dialect, writer, QUOTE_MINIMAL
from signedcentrality.clustering import Path
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

	makedirs(Path.RES_PATH, exist_ok=True)
	makedirs(Path.GENERATED_RES_PATH, exist_ok=True)
	makedirs(Path.R_GENERATED_RES_PATH, exist_ok=True)

	root = Element(XMLKeys.ROOT)
	for row in paths:
		if len(row) < 2:
			raise IndexError("The list must contain tuples of two elements.")

		SubElement(root, XMLKeys.PATH, **{"" + str(XMLKeys.INPUT_FILE): row[0], "" + str(XMLKeys.RESULT_FILE): row[1]})

	tree = ElementTree(root)
	tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def _set_float_param(param):
	"""
	Set the right number of digits for float parameters

	:param param: The float parameter
	:return: The formatted parameter
	"""

	param = str(round(float(str(param)), 4)).split('.')
	if len(param) == 1:
		param = param[0] + '.0000'
	else:
		param = param[0] + '.' + param[1] + ''.join(['0' for _ in range(max(0, 4 - len(param[1])))])

	return param


def _get_path_from_descr(base_path, **kwargs):
	"""
	Compute input file paths for the default sample dataset

	This function computes the path of the files containing the input graphs on the basis of the graph description.
	These paths are computed from the descriptions of the graphs in the CSV file containing the classes of solutions for the cluterring of these files.
	The description is given as a set of key/value pairs defining the characteristics of the graph.

	:param base_path: Path to the directory containing the input files.
	:type base_path: str
	:param kwargs: Description of the graph
	:type kwargs: dict
	:return: The path
	:rtype: dict
	"""

	return base_path.rstrip('/') + "/n={}_k={}_dens={}/propMispl={}/propNeg={}/network={}".format(kwargs['n'], kwargs['k'], _set_float_param(kwargs['d']), _set_float_param(kwargs['prop_mispl']), _set_float_param(kwargs['prop_neg']), kwargs['network_no'])


def read_class_results_csv(path, description_processing_function=_get_path_from_descr, base_path=None):
	"""
	Get the classes from a CSV file

	The path of the input file of each class is computed using the description of the graph, which is the first column of each row of the CSV file.
	The function to use to process it is given a a parameter.
	The default value of this parameter is the function that is used with the default dataset.

	:param path: Path
	:type path: str
	:param base_path: Base path to the input files which the full path is computed by this function
	:type base_path: str
	:param description_processing_function: Function to use to compute the path from the description
	:type description_processing_function: function
	:return: the results
	:rtype: tuple
	"""

	if base_path is None:
		base_path = Path.DEFAULT_SAMPLE_INPUTS_PATH

	data = read_csv(path, remove_headers=False)
	processed_data = {}
	
	for i in range(1, len(data)):
		parameters = ','.join(['='.join([param.split('=')[0].replace('.', '_'), param.split('=')[1]]) for param in str(data[i][0]).split(',')])
		str_func = str(description_processing_function.__name__) + '(base_path="' + base_path + '", ' + parameters + ')'
		dir_path = eval(str_func)
		# data[i][0] = dir_path

		processed_data = {**processed_data, dir_path: {
			ClassifierMode.SOLUTIONS_NUMBER: int(data[i][1]),
			ClassifierMode.SINGLE_SOLUTION: bool(data[i][2]),
			ClassifierMode.CLASSES_NUMBER: int(data[i][3]),
			ClassifierMode.SINGLE_CLASS: bool(data[i][4])
		}}

	return processed_data


def count_classes(file_path: str):
	"""
	Count class number in text files

	Text files must contain a number per line, which represent the class id for a node of the graph.
	This function counts the number of these class ids.

	:param file_path: path of the file
	:type file_path: str
	:return: class number
	:rtype: int
	"""

	class_ids = []
	class_number = 0

	with open(file_path, 'r') as file:

		print(file_path)

		for line in file.readline():
			id = None

			try:
				id = int(line.strip())
			except ValueError:
				continue

			if id not in class_ids:
				class_number += 1

			class_ids.append(id)

	return class_number


def load_data(training_data_directory_path: str, target_directory_path: str = None, input_files_paths_xml_file: str = None):
	"""
	Load dataset to train and test a Classifier.

	The directory tree must be the same into directories containing training files and target files.
	It is used to associate input files to target files.
	However, file names can be different. There is one result file for each optimal clustering.
	There can be extra subdirectories at the end of the path for result files.

	This function is designed to to create train/test datasets.
	However, it also can be used to load graphs to classify, if the path structure is the same.

	The parameter target_directory_path must be set only if this method is used to compute a training dataset.
	If the computed dataset is real data to classify, classifying methods don't need target data.

	:param training_data_directory_path: Path of training dataset
	:type training_data_directory_path: str
	:param target_directory_path: Path of target dataset
	:type target_directory_path: str
	:param input_files_paths_xml_file: Path to the XML file to write
	:type input_files_paths_xml_file: str
	:return: the loaded and parsed data in a tuple containing training data and target data
	"""

	# Load target data
	if input_files_paths_xml_file is None:
		input_files_paths_xml_file = Path.GENERATED_XML_PATHS_FILE

	targets = {}

	if target_directory_path is not None:
		for (dir_path, dir_names, file_names) in walk(target_directory_path):
			for file_name in file_names:
				targets = {**targets, **read_class_results_csv("/".join([dir_path, file_name]))}

	# # For tests:
	# for key, val in targets.items():
	# 	print(key)
	# 	for k, v in val.items():
	# 		print('\t', k, ": ", v, sep='')

	# Load input data:

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
			(Path.R_GENERATED_RES_PATH + input_file_path.replace(training_data_directory_path, '')).replace(
				Path.GRAPHML_EXT,
				Path.XML_EXT)  # Path to R generated file containing results for input file.
		] for input_file_path in input_files_paths
	]

	print('input_files_paths_xml_file:', input_files_paths_xml_file)

	write_xml(input_files_paths_xml_file, file_paths)

	# Compute the descriptors :
	system(
		Path.R_SCRIPT + " " +  # Path to the script to run
		Path.RES_PATH + " " +  # Current working directory of this script
		input_files_paths_xml_file  # Path to the XML file containing the paths to files whose descriptors must be computed, and files to write the computed descriptors.
	)

	training_data = {}
	for io_paths in file_paths:
		input_file_path = io_paths[0]
		result_file_path = io_paths[1]

		xml_results = read_xml(result_file_path)
		centralities = compute_centralities_mean_stddev(read_graph(input_file_path))

		training_data = {**training_data, input_file_path: {**xml_results, **centralities}}

	# Compute target data :
	target_data = {
		ClassifierMode.SINGLE_CLASS: {},
		ClassifierMode.CLASSES_NUMBER: {},
		ClassifierMode.SINGLE_SOLUTION: {},
		ClassifierMode.SOLUTIONS_NUMBER: {}
	}

	for input_file_path in training_data.keys():
		target_data[ClassifierMode.SINGLE_CLASS] = {**target_data[ClassifierMode.SINGLE_CLASS], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SINGLE_CLASS]}
		target_data[ClassifierMode.CLASSES_NUMBER] = {**target_data[ClassifierMode.CLASSES_NUMBER], input_file_path: targets[dirname(input_file_path)][ClassifierMode.CLASSES_NUMBER]}
		target_data[ClassifierMode.SINGLE_SOLUTION] = {**target_data[ClassifierMode.SINGLE_SOLUTION], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SINGLE_SOLUTION]}
		target_data[ClassifierMode.SOLUTIONS_NUMBER] = {**target_data[ClassifierMode.SOLUTIONS_NUMBER], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SOLUTIONS_NUMBER]}

	return training_data, target_data


class Classifier:
	"""
	This class computes the number of classes of solutions for a clustering.
	"""

	def __init__(self, classifier: SVC, mode: ClassifierMode, training_vectors=None, target_values=None):
		"""
		Creates a newly allocated Classifier object.

		:param classifier: SVC classifier which must be used as classifier.
		:param training_vectors: Input descriptors to train the classifier.
		:param target_values: Class numbers of input descriptors.
		"""

		if training_vectors is None or target_values is None:
			print(
				'The parameters training_vectors and target_values are not set.',
				'The attributes data and target must be set before the training.',
				sep='\n', file=stderr
			)

		self.data = training_vectors
		self.target = target_values

		self.__classifier = classifier
		self.__mode = mode
		self.__accuracy_score = None
		self.__precision_score = None
		self.__recall_score = None
		self.__accuracy_list = []
		self.__precision_list = []
		self.__recall_list = []

		self.__train_data, self.__test_data, self.__train_target, self.__test_target = train_test_split(self.data, self.target, test_size=.3)

	@property
	def mode(self):
		return self.__mode

	@property
	def accuracy_score(self):
		return self.__accuracy_score

	@property
	def precision_score(self):
		return self.__precision_score

	@property
	def recall_score(self):
		return self.__recall_score

	@property
	def accuracy_list(self):
		return self.__accuracy_list

	@property
	def precision_list(self):
		return self.__precision_list

	@property
	def recall_list(self):
		return self.__recall_list

	@property
	def train_data(self):
		return self.__train_data

	@property
	def test_data(self):
		return self.__test_data

	@property
	def train_target(self):
		return self.__train_target

	@property
	def test_target(self):
		return self.__test_target

	def train(self):
		"""
		Train and test the classifier

		The classifier in trained with 70% of training data and 30% of test data.
		Data set is randomly divided.

		:param mode: Mode for which the classifier must be set
		:return: means for all accuracy, precision and recall scores
		"""

		self.__classifier.fit(
			self.__train_data,  # Lists of descriptors
			self.__test_target  # List of results for each list of descriptors
		)

		for test_data in self.__test_data:
			predicted_test_target = self.__classifier.predict(test_data)
			self.__accuracy_list.append(accuracy_score(self.__test_target, predicted_test_target))
			self.__precision_list.append(precision_score(self.__test_target, predicted_test_target))
			self.__recall_list.append(recall_score(self.__test_target, predicted_test_target))

		self.__accuracy_score = mean(self.__accuracy_list)
		self.__precision_score = mean(self.__precision_list)
		self.__recall_score = mean(self.__recall_list)

		return {'accuracy': self.__accuracy_score, 'precision': self.__precision_score, 'recall': self.__recall_score}

	def get_class_number(self, graph):
		predicted_result = self.__classifier.predict()
		print(predicted_result)

		return predicted_result
