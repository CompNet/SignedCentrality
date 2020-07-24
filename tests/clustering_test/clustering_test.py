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
from signedcentrality.clustering import SVCKernel, ClassifierMode
from signedcentrality.clustering.classifier import Classifier, load_data, format_train_test_data
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
	chdir("/".join([getcwd(), Path.TESTS_RES_PATH]))
	Path.load()

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

	data = format_train_test_data(training_data, target_data)

	# # Tests :
	# for mode, value in data.items():
	# 	print(mode)
	# 	for data, v in value.items():
	# 		print('\t', data, sep='')
	#
	# 		if data == ClassifierData.INPUT:
	# 			for descriptors in v:
	# 				for descriptor in descriptors:
	# 					print('\t\t', descriptor, sep='')
	# 				print()
	#
	# 		else:
	# 			for target in v:
	# 				print('\t\t', target, sep='')

	main_data = (data, training_data, target_data)

	return main_data


def generic_classifier_test(params: dict, mode: ClassifierMode, data, print_result=True):
	"""
	Generic test for classifier or regressor

	:param params: parameters for the classifier or regressor
	:param mode: Classifier mode
	:param data: data to use
	:param print_result: if True, the results are printed
	:return: the results
	"""

	separator = "".join(['-' for _ in range(79)])

	svm = None
	if mode == ClassifierMode.SOLUTIONS_NUMBER or mode == ClassifierMode.CLASSES_NUMBER:
		svm = SVR(**params)
	else:
		svm = SVC(**params)

	if print_result:
		print()
		print(separator)
		print(str(mode).replace(str(mode.__class__).replace("<enum '", "").replace("'>", "") + '.', ''), params, sep='\t')
		print(separator)

	classifier = Classifier(svm, mode, *(data[mode].values()))
	result = classifier.train()

	if print_result:
		print(result)
		print(separator)
		print()

	return result


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		self.data, self.training_data, self.target_data = init_svm()

		self.main_params_list = [
			{'kernel': SVCKernel.LINEAR},
			{'kernel': SVCKernel.RBF, 'gamma': 'scale'},  # 'scale' is default value.
			{'kernel': SVCKernel.RBF, 'gamma': 'auto'},
			{'kernel': SVCKernel.POLY, 'gamma': 'scale'},  # 'scale' is default value.
			{'kernel': SVCKernel.POLY, 'gamma': 'auto', 'max_iter': 100_000},  # Because default value is too long to compute.
			{'kernel': SVCKernel.POLY, 'gamma': 'auto', 'max_iter': 1_000_000},  # Because default value is too long to compute.
			{'kernel': SVCKernel.SIGMOID, 'gamma': 'scale'},  # 'scale' is default value.
			{'kernel': SVCKernel.SIGMOID, 'gamma': 'auto'}
		]  # These parameters are the main ones, those which cover the kernel settings.

		additional_parameters = {
			'tol': 1e-1,  # 1e-3 is default value.
			'shrinking': False,  # True is default value.
		}  # These parameters must be combined with all others.

		additional_svc_parameters = {
			'probability': True,  # False is default value.
			'decision_function_shape': 'ovo'  # 'ovr' is default value.
		}  # These parameters must be combined with all others, but only for a SVC classifier.

		additional_svr_parameters = {
		}  # These parameters must be combined with all others, but only for a SVR regressor.

		for key, value in additional_parameters.items():
			self.main_params_list = [
				*self.main_params_list,
				*[{**params, key: value} for params in self.main_params_list]
			]

		self.svc_params_list = [*self.main_params_list]
		self.svr_params_list = [*self.main_params_list]

		for key, value in additional_svc_parameters.items():
			self.svc_params_list = [
				*self.svc_params_list,
				*[{**params, key: value} for params in self.svc_params_list]
			]

		for key, value in additional_svr_parameters.items():
			self.svr_params_list = [
				*self.svr_params_list,
				*[{**params, key: value} for params in self.svr_params_list]
			]

	def test_load_data(self):
		"""
		Test if the sets have the same keys
		"""

		self.assertEqual(len(self.training_data.keys()), len(self.target_data[ClassifierMode.SINGLE_CLASS].keys()))
		self.assertEqual(self.training_data.keys(), self.target_data[ClassifierMode.SINGLE_CLASS].keys())
		self.assertEqual(len(self.training_data.keys()), len(self.target_data[ClassifierMode.CLASSES_NUMBER].keys()))
		self.assertEqual(self.training_data.keys(), self.target_data[ClassifierMode.CLASSES_NUMBER].keys())
		self.assertEqual(len(self.training_data.keys()), len(self.target_data[ClassifierMode.SINGLE_SOLUTION].keys()))
		self.assertEqual(self.training_data.keys(), self.target_data[ClassifierMode.SINGLE_SOLUTION].keys())
		self.assertEqual(len(self.training_data.keys()), len(self.target_data[ClassifierMode.SOLUTIONS_NUMBER].keys()))
		self.assertEqual(self.training_data.keys(), self.target_data[ClassifierMode.SOLUTIONS_NUMBER].keys())

	def test_classifier_single_solution(self):

		params_list = [
			*self.svc_params_list
		]  # Additional parameters can be added here only for this test case.

		for params in params_list:
			if params['kernel'] == 'linear' and 'probability' in params.keys() and params['probability']:
				params = {**params, 'max_iter': 1_000_000}

			with self.subTest(**params):
				generic_classifier_test(params, ClassifierMode.SINGLE_SOLUTION, self.data)

	def test_classifier_single_class(self):

		params_list = [
			*self.svc_params_list
		]  # Additional parameters can be added here only for this test case.

		for params in params_list:
			with self.subTest(**params):
				generic_classifier_test(params, ClassifierMode.SINGLE_CLASS, self.data)

	def test_classifier_solutions_number(self):

		params_list = [
			*self.svr_params_list
		]  # Additional parameters can be added here only for this test case.

		for params in params_list:
			with self.subTest(**params):
				generic_classifier_test(params, ClassifierMode.SOLUTIONS_NUMBER, self.data)

	def test_classifier_classes_number(self):

		params_list = [
			*self.svr_params_list
		]  # Additional parameters can be added here only for this test case.

		for params in params_list:
			with self.subTest(**params):
				generic_classifier_test(params, ClassifierMode.CLASSES_NUMBER, self.data)


if __name__ == '__main__':
	unittest.main()
