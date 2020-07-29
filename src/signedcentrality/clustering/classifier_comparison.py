#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains tests to compare classifier parameters.

The dataset "Space of optimal solutions of the Correlation Clustering problem" (Version 3) is used for comparisons.

.. note: Arinik, Nejat; Labatut, Vincent (2019): Space of optimal solutions of the Correlation Clustering problem. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3
"""

from csv import writer, QUOTE_MINIMAL
from os import getcwd, chdir, makedirs
from os.path import dirname
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.svm import SVC, SVR
from signedcentrality.clustering import SVCKernel, ClassifierMode, ClassifierData, Path
from signedcentrality.clustering.classifier import Classifier, initialize_data


def _write_result_file(file_base_name, *results):
	"""
	Creates a CSV file containing the results

	:param results: results to write
	:type results: list
	:param path: the path of the CSV file
	:type path: str
	"""

	file_base_name = file_base_name.replace(Path.DEFAULT_SAMPLE_INPUTS_PATH, Path.PREDICTED_RESULTS)
	path = Path.GENERATED_RES_PATH + '/' + file_base_name + Path.CSV_EXT

	split_path = dirname(file_base_name).split('/')
	if len(split_path) > 0:
		tmp_path = '' + Path.GENERATED_RES_PATH
		for directory in split_path:
			tmp_path += '/' + directory
			makedirs(tmp_path, exist_ok=True)

	with open(path, 'w') as file:
		csv_writer = writer(file, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL)
		csv_writer.writerow(results)


class ClassifierComparator:
	def __init__(self, combine_parameters=True, tasks=None, compute_descriptors=True):
		"""
		Initialize a new ClassifierComparator object

		:param combine_parameters: if True, parameters are combined
		:param tasks: prediction tasks
		:param compute_descriptors: if True, descriptors are computed before the training
		"""

		self.__data, self.__graph_ids, self.__input, self.__target = initialize_data(compute_descriptors=compute_descriptors)
		self.__train, self.__validation, self.__test = self.__data
		self.__train_graph_ids, self.__validation_graph_ids, self.__test_graph_ids = self.__graph_ids

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

		if combine_parameters:
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

		else:
			for key, value in additional_parameters.items():
				self.__main_params_list.append({key: value})

			self.__svc_params_list = [*self.__main_params_list]
			self.__svr_params_list = [*self.__main_params_list]

			for key, value in additional_svc_parameters.items():
				self.__svc_params_list.append({key: value})

			for key, value in additional_svr_parameters.items():
				self.__svr_params_list.append({key: value})

		self.tasks = {
			ClassifierMode.SINGLE_SOLUTION: self.__svc_params_list,
			ClassifierMode.SINGLE_CLASS: self.__svc_params_list,
			ClassifierMode.SOLUTIONS_NUMBER: self.__svr_params_list,
			ClassifierMode.CLASSES_NUMBER: self.__svr_params_list
		}

		if tasks is not None:
			if isinstance(tasks, ClassifierMode):
				self.tasks = {tasks: self.tasks[tasks]}
			else:
				tasks_dict = {}
				for task in tasks:
					tasks_dict = {
						**tasks_dict,
						task: self.tasks[task]
					}

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
	def generic_classifier_comparison(params: dict, mode: ClassifierMode, training_data, validation_data, training_data_graph_ids=None, validation_data_graph_ids=None, print_result=False, print_progress=False):
		"""
		Generic comparison for classifier or regressor

		:param params: parameters for the classifier or regressor
		:param mode: Classifier mode
		:param validation_data: data to use to validate training
		:param training_data: data to use to train
		:param print_result: if True, the results are printed
		:param training_data_graph_ids: ids of train graphs
		:param validation_data_graph_ids: ids of validation graphs
		:param print_progress: if True, the method train() prints its progress
		:return: the results
		"""

		separator = "".join(['-' for _ in range(79)])

		svm = None
		if mode == ClassifierMode.SOLUTIONS_NUMBER or mode == ClassifierMode.CLASSES_NUMBER:
			svm = SVR(**params)
		else:
			svm = SVC(**params)

		score_file_name = str(mode).replace(str(mode.__class__).replace("<enum '", "").replace("'>", "") + '.', '') + "_-_" + "_".join(["=".join([str(key), str(value)]) for key, value in params.items()])

		if print_result:
			print()
			print(separator)
			# print(str(mode).replace(str(mode.__class__).replace("<enum '", "").replace("'>", "") + '.', ''), params, sep='\t')
			print(score_file_name)
			print(separator)

		classifier = Classifier(svm, mode, *(training_data[mode].values()), *(validation_data[mode].values()), training_data_graph_ids, validation_data_graph_ids)
		report, result = classifier.train(detailed=True, print_progress=print_progress)

		if print_result:
			print(separator)
			print(report)
			print()

		_write_result_file(score_file_name, report)

		for graph_path, predicted_result in result.items():
			_write_result_file(graph_path, report)

		return result

	def compare_classifiers(self, print_progress=False):
		"""
		Compare training results for all prediction tasks.
		"""

		tasks = self.tasks

		tests_number = 0
		for task, params_list in tasks.items():
			for _ in params_list:
				tests_number += 1

		test_counter = 0
		progress = 0
		if print_progress:
			print('Number of tests : {}'.format(tests_number), 'Comparison progress : ', '0 %', sep='\n')

		for task, params_list in tasks.items():
			for params in params_list:
				if print_progress:
					test_counter += 1

				ClassifierComparator.generic_classifier_comparison(params, task, self.__train, self.__validation, self.__train_graph_ids, self.__validation_graph_ids, print_progress=print_progress)

				if print_progress:
					# progress = (test_counter / tests_number) * 100
					progress = round((test_counter / tests_number) * 100, 2)
					print(progress, '%')
					print()
					print()

	def classifier_test(self, classifier: Classifier, print_result=False):
		"""
		Test results for the selected Classifier

		:param classifier: the selected Classifier
		:param print_result: if True, the results are printed
		"""

		mode = classifier.mode

		data, target = self.__test[mode][ClassifierData.INPUT], self.__test[mode][ClassifierData.TARGET]
		result = None

		predicted = []
		for d in data:
			predicted.append(classifier.predict(d))

		if isinstance(classifier.classifier, SVC):
			result = f1_score(target, predicted)
		else:  # if isinstance(classifier.classifier, SVR)
			result = mean_absolute_error(data, target)

		if print_result:
			print(result)

		_write_result_file('optimal_parameters_test', result)


if __name__ == '__main__':

	# Change the path of unit tests working directory:
	chdir("/".join([getcwd(), '../../../res']))
	Path.load()

	comparator = ClassifierComparator()
	# comparator = ClassifierComparator(compute_descriptors=False)  # If graph descriptors have been already computed.
	comparator.compare_classifiers(True)

