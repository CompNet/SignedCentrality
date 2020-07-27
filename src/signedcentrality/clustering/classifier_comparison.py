#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tests to compare classifier parameters.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for comparisons.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

from csv import writer, QUOTE_MINIMAL
from os import getcwd, chdir
from sklearn.svm import SVC, SVR
from signedcentrality.clustering import SVCKernel, ClassifierMode
from signedcentrality.clustering.classifier import Classifier, initialize_data
from tests.clustering_test import Path


def _write_result_file(file_base_name, *results):
	"""
	Creates a CSV file containing the results

	:param results: results to write
	:type results: list
	:param path: the path of the CSV file
	:type path: str
	"""

	path = Path.GENERATED_RES_PATH + '/' + file_base_name + Path.CSV_EXT

	with open(path, 'w') as file:
		csv_writer = writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
		csv_writer.writerow(results)


class ClassifierComparator:
	def __init__(self):
		"""
		Initialize a new ClassifierComparator object
		"""

		self.__data, self.__input, self.__target = initialize_data()
		self.__train, self.__validation, self.__test = self.__data

		self.__main_params_list = [
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
			# 'probability': True,  # False is default value.
			'decision_function_shape': 'ovo'  # 'ovr' is default value.
		}  # These parameters must be combined with all others, but only for a SVC classifier.

		additional_svr_parameters = {
		}  # These parameters must be combined with all others, but only for a SVR regressor.

		for key, value in additional_parameters.items():
			self.__main_params_list = [
				*self.__main_params_list,
				*[{**params, key: value} for params in self.__main_params_list]
			]

		self.__svc_params_list = [*self.__main_params_list]
		self.__svr_params_list = [*self.__main_params_list]

		for key, value in additional_svc_parameters.items():
			self.__svc_params_list = [
				*self.__svc_params_list,
				*[{**params, key: value} for params in self.__svc_params_list]
			]

		for key, value in additional_svr_parameters.items():
			self.__svr_params_list = [
				*self.__svr_params_list,
				*[{**params, key: value} for params in self.__svr_params_list]
			]

	@property
	def input(self):
		return self.__input

	@property
	def target(self):
		return self.__target

	@property
	def train(self):
		return self.__train

	@property
	def validation(self):
		return self.__validation

	@property
	def test(self):
		return self.__test

	@staticmethod
	def generic_classifier_comparison(params: dict, mode: ClassifierMode, training_data, validation_data, print_result=False):
		"""
		Generic comparison for classifier or regressor

		:param params: parameters for the classifier or regressor
		:param mode: Classifier mode
		:param validation_data: data to use to validate training
		:param training_data: data to use to train
		:param print_result: if True, the results are printed
		:return: the results
		"""

		separator = "".join(['-' for _ in range(79)])

		svm = None
		if mode == ClassifierMode.SOLUTIONS_NUMBER or mode == ClassifierMode.CLASSES_NUMBER:
			svm = SVR(**params)
		else:
			svm = SVC(**params)

		result_file_name = str(mode).replace(str(mode.__class__).replace("<enum '", "").replace("'>", "") + '.', '') + "_-_" + "_".join(["=".join([str(key), str(value)]) for key, value in params.items()])

		if print_result:
			print()
			print(separator)
			# print(str(mode).replace(str(mode.__class__).replace("<enum '", "").replace("'>", "") + '.', ''), params, sep='\t')
			print(result_file_name)
			print(separator)

		classifier = Classifier(svm, mode, *(training_data[mode].values()), *(validation_data[mode].values()))
		result = classifier.train()

		if print_result:
			print(result)
			print(separator)
			print()

			_write_result_file(result_file_name, result)

		return result

	def compare_classifiers(self):
		"""
		Compare training results for all prediction tasks.
		"""

		tasks = {
			ClassifierMode.SINGLE_SOLUTION: self.__svc_params_list,
			ClassifierMode.SINGLE_CLASS: self.__svc_params_list,
			ClassifierMode.SOLUTIONS_NUMBER: self.__svr_params_list,
			ClassifierMode.CLASSES_NUMBER: self.__svr_params_list
		}

		for task, params_list in tasks.items():
			for params in params_list:
				ClassifierComparator.generic_classifier_comparison(params, task, self.__train, self.__validation)

	def classifier_test(self, classifier):
		"""
		Test results for the selected Classifier

		:param classifier: the selected Classifier
		"""

		pass


if __name__ == '__main__':

	# Change the path of unit tests working directory:
	chdir("/".join([getcwd(), '../../../res']))
	Path.load()

	comparator = ClassifierComparator()
	comparator.compare_classifiers()

