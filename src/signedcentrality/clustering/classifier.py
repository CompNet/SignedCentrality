#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a classifier which uses centralities computed in centrality package.
"""
from collections import OrderedDict
from os import walk, makedirs, system
from os.path import dirname, splitext, basename
from statistics import mean, stdev
from xml.etree.ElementTree import parse, Element, ElementTree, SubElement
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# noinspection PyProtectedMember
from signedcentrality._utils.utils import *
from signedcentrality.centrality import CentralityMeasure
from signedcentrality.centrality.degree_centrality import PNCentrality
from signedcentrality.centrality.eigenvector_centrality import EigenvectorCentrality
from signedcentrality.clustering import Path, ClassifierTraining
from signedcentrality.clustering import XMLKeys, ClassifierMode, ClassifierData


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


def read_xml(path, additional_descriptors=None):
	"""
	Read an XML file

	This file must contain results written by the R script.

	If additional_descriptors are set, the file will contain its descriptors and the additional ones.

	:param path: path of the XML file
	:type path: str
	:param additional_descriptors: descriptors to add to the XML file
	:type additional_descriptors: dict
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

	if additional_descriptors is None:
		return results

	results = {**results, **additional_descriptors}

	for descriptor_name, descriptor_value in additional_descriptors.items():
		SubElement(root, XMLKeys.DESCRIPTOR, **{"" + str(XMLKeys.TYPE): str(descriptor_name), "" + str(XMLKeys.VALUE): str(descriptor_value)})

	xml_tree = ElementTree(root)
	xml_tree.write(path, encoding='utf-8', xml_declaration=True)

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
			ClassifierMode.SINGLE_SOLUTION: bool(int(data[i][2])),
			ClassifierMode.CLASSES_NUMBER: int(data[i][3]),
			ClassifierMode.SINGLE_CLASS: bool(int(data[i][4]))
		}}

	return processed_data


def load_data(training_data_directory_path: str, target_directory_path: str = None, input_files_paths_xml_file: str = None, compute_descriptors=True):
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

	The parameter compute_descriptors must be used only if descriptors have already been computed.

	:param training_data_directory_path: Path of training dataset
	:type training_data_directory_path: str
	:param target_directory_path: Path of target dataset
	:type target_directory_path: str
	:param input_files_paths_xml_file: Path to the XML file to write
	:type input_files_paths_xml_file: str
	:param compute_descriptors: If  false, descriptors are not computed.
	:type compute_descriptors: bool
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

	if compute_descriptors:
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

		descriptors = None
		if compute_descriptors:
			centralities = compute_centralities_mean_stddev(read_graph(input_file_path))
			descriptors = read_xml(result_file_path, centralities)

		else:
			descriptors = read_xml(result_file_path)

		training_data = {**training_data, input_file_path: descriptors}

	# Compute target data :
	target_data = {
		ClassifierMode.SINGLE_CLASS: {},
		ClassifierMode.CLASSES_NUMBER: {},
		ClassifierMode.SINGLE_SOLUTION: {},
		ClassifierMode.SOLUTIONS_NUMBER: {}
	}

	del_keys = []  # If training files don't have target, they must be removed.
	for input_file_path in training_data.keys():  # If target data doesn't have training file, it won't be added.
		if dirname(input_file_path) not in targets:
			del_keys.append(input_file_path)  # Add key to removed files list.
			continue

		target_data[ClassifierMode.SINGLE_CLASS] = {**target_data[ClassifierMode.SINGLE_CLASS], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SINGLE_CLASS]}
		target_data[ClassifierMode.CLASSES_NUMBER] = {**target_data[ClassifierMode.CLASSES_NUMBER], input_file_path: targets[dirname(input_file_path)][ClassifierMode.CLASSES_NUMBER]}
		target_data[ClassifierMode.SINGLE_SOLUTION] = {**target_data[ClassifierMode.SINGLE_SOLUTION], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SINGLE_SOLUTION]}
		target_data[ClassifierMode.SOLUTIONS_NUMBER] = {**target_data[ClassifierMode.SOLUTIONS_NUMBER], input_file_path: targets[dirname(input_file_path)][ClassifierMode.SOLUTIONS_NUMBER]}

	for key in del_keys:
		del training_data[key]  # Remove training data which don't have target.

	return training_data, target_data


def initialize_data(compute_descriptors=True):
	"""
	Initialize dataset to train the classifier

	:param compute_descriptors: If  false, descriptors are not computed.
	:type compute_descriptors: bool
	:return: the data
	"""

	training_data, target_data = load_data(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.DEFAULT_SAMPLE_RESULTS_PATH, compute_descriptors=compute_descriptors)

	paths = list(OrderedDict(training_data).keys())  # There are the same paths for training data and target data.
	inputs = [
				[
					value for key, value in OrderedDict(training_data[path]).items()
				] for path in paths
			]

	target_indexes = [
		indexes for indexes in range(len(paths))
	]  # Indexes are used because the datasets must have results for the same inputs in the same order for each prediction task.

	# Now, inputs and target_indexes are ordered such that both of them are alphabetically sorted by the name of the paths. They don't contain these paths.

	train_data, validation_test_data, train_target_indexes, validation_test_target_indexes = train_test_split(inputs, target_indexes, test_size=.3)
	validation_data, test_data, validation_target_indexes, test_target_indexes = train_test_split(validation_test_data, validation_test_target_indexes, test_size=.5)

	data = {
		prediction_task: {
			mode: {
				ClassifierData.INPUT: data[0],
				ClassifierData.TARGET: [
					int(target_data[mode][paths[i]]) for i in data[1]  # int(), because classifier uses int values.
				],
			} for mode in [
				ClassifierMode.SINGLE_CLASS,
				ClassifierMode.CLASSES_NUMBER,
				ClassifierMode.SINGLE_SOLUTION,
				ClassifierMode.SOLUTIONS_NUMBER
			]
		} for prediction_task, data in {
			ClassifierTraining.TRAIN: (train_data, train_target_indexes),
			ClassifierTraining.VALIDATION: (validation_data, validation_target_indexes),
			ClassifierTraining.TEST: (test_data, test_target_indexes)
		}.items()
	}

	graph_ids = {
		prediction_task: [
			paths[i] for i in indexes_list
		] for prediction_task, indexes_list in {
			ClassifierTraining.TRAIN: train_target_indexes,
			ClassifierTraining.VALIDATION: validation_target_indexes,
			ClassifierTraining.TEST: test_target_indexes
		}.items()
	}

	return (
		(
			data[ClassifierTraining.TRAIN],
			data[ClassifierTraining.VALIDATION],
			data[ClassifierTraining.TEST]
		),
		(
			graph_ids[ClassifierTraining.TRAIN],
			graph_ids[ClassifierTraining.VALIDATION],
			graph_ids[ClassifierTraining.TEST]
		),
		training_data,
		target_data
	)


class Classifier:
	"""
	This class computes the number of solutions and classes of solutions for the Optimal Clustering Problem.

	It also computes if there are one or several solutions or classes.
	It uses a classification or a regression, depending on the chosen mode.
	Classification is computed using the classifier sklearn.svm.SVC.
	Regression is computed using the classifier sklearn.svm.SVR.

	There are four mode which define what is computed :
	- ClassifierMode.SINGLE_CLASS (classification): computes if there are one or several classes.
	- ClassifierMode.CLASSES_NUMBER (regression): computes the number of classes of solutions.
	- ClassifierMode.SINGLE_SOLUTION (classification): computes if there are one or several solutions.
	- ClassifierMode.SOLUTIONS_NUMBER (regression): computes the number of solutions.

	It also gives information about the training using SciKit Learn metrics.
	There are different metrics for classification and regression.
	In this class, metrics documentation give information gathered on SciKit learn documentation.
	"""

	def __init__(self, classifier, mode: ClassifierMode, train_data, train_target, validation_data, validation_target, train_graph_ids=None, validation_graph_ids=None):
		"""
		Creates a newly allocated Classifier object.

		The optional last parameters represent ids of graphs which are used for the training.
		So, the results of the validation can be associated with the graphs.

		:param classifier: SVC classifier which must be used as classifier.
		:param mode: Classifier mode
		:param train_data: input data to train the classifier
		:param train_target: target data to train the classifier
		:param validation_data: input data to validate the training of the classifier
		:param validation_target: target data to validate the training of the classifier
		:param train_graph_ids: ids of train graphs
		:param validation_graph_ids: ids of validation graphs
		"""

		self.__train_data = train_data
		self.__validation_data = validation_data
		self.__train_target = train_target
		self.__validation_target = validation_target
		self.__train_graph_ids = train_graph_ids
		self.__validation_graph_ids = validation_graph_ids

		self.__classifier = classifier
		self.__mode = mode

		# Classification metrics for training :
		self.__f1_score = None
		"""
		Balanced F-score

		According to the SciKit Learn Documentation, 

			"The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal."
		"""

		# Regression metrics for training :
		self.__mean_squared_error = None
		"""
		Mean absolute error

		According to the SciKit Learn Documentation, 

			"The mean_squared_error function computes mean squared error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss."
		"""

	@property
	def classifier(self):
		return self.__classifier

	@property
	def train_data(self):
		return self.__train_data

	@property
	def train_target(self):
		return self.__train_target

	@property
	def validation_data(self):
		return self.__validation_data

	@property
	def validation_target(self):
		return self.__validation_target

	@property
	def mode(self):
		return self.__mode

	@property
	def f1_score(self):
		return self.__f1_score

	@property
	def mean_squared_error(self):
		return self.__mean_squared_error

	@property
	def test_data(self):
		return self.__validation_data

	@property
	def test_target(self):
		return self.__validation_target

	def __compute_classification_metrics(self, test_target, predicted_test_target):
		"""
		Compute the metrics for a classification

		:param test_target: right values
		:param predicted_test_target: values which have been predicted by the classifier
		:return: a report showing the metrics
		"""

		self.__f1_score = f1_score(test_target, predicted_test_target, zero_division=0)  # 0 is the worst score.
		report = self.__f1_score

		return report

	def __compute_regression_metrics(self, test_target, predicted_test_target):
		"""
		Compute the metrics for a regression

		:param test_target: right values
		:param predicted_test_target: values which have been predicted by the regression
		:return: a report showing the metrics
		"""

		self.__mean_squared_error = mean_squared_error(test_target, predicted_test_target)
		report = self.__mean_squared_error

		return report

	def train(self, detailed=False, print_progress=False):
		"""
		Train and test the classifier

		The classifier in trained with 70% of training data and 30% of test data.
		Data set is randomly divided.

		:param print_progress: if True, the method prints its progress
		:param detailed: if True, results contain more details
		:return: means for all accuracy, precision and recall scores
		"""

		if print_progress:
			print('\tTraining in progress ...')

		self.__classifier.fit(
			self.__train_data,  # Lists of descriptors
			self.__train_target  # List of results for each list of descriptors
		)

		if print_progress:
			print('\tTraining has been done successfully.')

		predicted_validation_target_list = []
		length = len(self.__validation_data)

		test_counter = 0
		progress = 0
		if print_progress:
			print('\tNumber of tests : {}'.format(length), '\tValidation progress : ', '\t0 %', sep='\n')
			print()

		for i in range(length):
			if print_progress:
				test_counter += 1

			validation_data = self.__validation_data[i]
			predicted_validation_target = self.__classifier.predict([validation_data])
			predicted_validation_target_list.append(predicted_validation_target)

			if print_progress:
				new_progress = round((test_counter / length) * 100, 0)
				if new_progress > progress:
					progress = new_progress
					print('\t', progress, '%', sep='')

		if print_progress:
			print()

		report = None
		if isinstance(self.__classifier, SVC):
			report = self.__compute_classification_metrics(self.__validation_target, predicted_validation_target_list)
		else:  # if isinstance(self.__classifier, SVR)
			report = self.__compute_regression_metrics(self.__validation_target, predicted_validation_target_list)

		results = {}
		if detailed:
			for i in range(length):
				results = {
					**results,
					self.__validation_graph_ids[i]: predicted_validation_target_list[i]
				}

			return report, results

		return report

	def predict(self, descriptors: list):
		"""
		Predict the result for the descriptors of a graph.

		Descriptors must be the same as for the training.
		The predicted result is the mode which have been set in the constructor.
		To predict for another mode, one has to train another classifier.

		This function returns an integer value.

		:param descriptors: descriptors of a graph.
		:return: the predicted result
		"""

		predicted_result = self.__classifier.predict([descriptors])
		return predicted_result
