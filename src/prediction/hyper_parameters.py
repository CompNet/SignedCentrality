#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to hyper parameters tes and comparison.

@author: Virgile Sucal
"""
import os
from math import isnan
from os.path import abspath, dirname, join, isdir, exists, isfile
from sys import stderr
from time import time

from deprecated import deprecated
from sklearn import metrics
import consts
import sys
from collect.collect_graphics import generate_plot, generate_errorbar_plot, generate_boxplot_clean, \
    generate_boxplot_clean1, generate_std_boxplot, generate_std_violinplot
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics
from prediction.classification import perform_svc_classification
from prediction.regression import perform_linear_regression, perform_mlp_regression, perform_svr_regression
from prediction.classification import perform_svc_classification
from prediction.random_forest_classification import perform_random_forest_classification
from util import write_csv, ProgressBar, export_running_time
from path import get_csv_folder_path


def pop(dictionary):
    """
    Remove first key of a dictionary and its value and return both

    :param dictionary: the dictionary
    :return: first key and its value
    """

    key = [key for key in dictionary.keys()][0]
    value = dictionary[key]
    del dictionary[key]
    return key, value


def __initialize_hyper_parameters_sets(param_set=None, **parameters_range):
    """
    Compute all parameter combinations based on given parameter ranges

    :param param_set: set containing only one value for each parameter
    :param parameters_range: range of values for each parameter
    :return: all parameters set
    """

    if param_set is None:
        param_set = {}

    if len(parameters_range) == 0:
        # print(__define_output_file_name(param_pool))
        return [{**param_set}]
    param, values = pop(parameters_range)

    results = []
    for value in values:
        results = [
            *results,
            *__initialize_hyper_parameters_sets({**param_set, param: value}, **parameters_range)
        ]

    return results


def __initialize_graphic_data(best_param_set, results, results_ranges):
    """
    Initialise data for graphic functions

    :param best_param_set: best parameters set
    :param results: all parameters set
    :return: the data
    """

    data = {}

    for (metric_name, param_set, metric_value) in best_param_set:
        metric_range = None
        ok = True
        for results_range in results_ranges:
            if sum(results_range[metric_name]) / len(results_range[metric_name]) == metric_value:
                metric_range = results_range
                break
            # for k in param_set.keys():
            #     print(k, ":", results_range[k], "!=", param_set[k])
            #     if results_range[k] != param_set[k]:
            #         ok = False
            #         break
            # print()
            # if not ok:
            #     continue
            # metric_range = results_range[metric_name]
            # break
        # print('metric_range:', metric_range)
        metric_data = {param_name: [(param_set[param_name], metric_value, metric_range[metric_name])] for param_name in param_set.keys()}

        for i in range(len(results)):
            result = results[i]
            results_range = results_ranges[i]
            non_optimal_values = []
            for param, value in param_set.items():  # Don't use result, because it contains more columns than best_param_set.
                if value != result[param]:
                    non_optimal_values.append(param)
            if len(non_optimal_values) != 1:
                continue

            tested_param_name = non_optimal_values[0]
            metric_data[tested_param_name].append((result[tested_param_name], result[metric_name], results_range[metric_name]))

        data[metric_name] = {param_name: tuple(data_list) for param_name, data_list in metric_data.items()}

    return data


def print_parameters_comparison(param_name, param_values, metric_name, metric_values, metric_values_ranges, graphic_name):
    """
    Export graphic for all values of all hyper parameters

    :param param_name: name of the parameter
    :param param_values: values for the parameter
    :param metric_name: name of the metric
    :param metric_values: values for the metric
    :param metric_values_ranges: values ranges for the metric
    :param graphic_name: name of the graphic
    """

    param_name = str(param_name)
    metric_name = str(metric_name)
    generate_errorbar_plot(param_values, metric_values, metric_values_ranges, graphic_name, param_name.replace("_", " "), metric_name.replace("_", " "), print_title=False, dash_between_name_and_plot=True)
    generate_std_violinplot(param_values, metric_values_ranges, graphic_name, param_name.replace("_", " "), metric_name.replace("_", " "), print_title=False, dash_between_name_and_plot=True)


def print_parameters_comparisons(prediction_function_name, best_param_set, results, results_ranges, output):
    """
    Export graphic for all values of all hyper parameters

    :param prediction_function_name: name of the function that makes the predictions
    :param best_param_set: best parameters set
    :param results: all parameters set
    :param results_ranges: all parameters set with values ranges for the metric
    :param output: prediction task
    """

    data = __initialize_graphic_data(best_param_set, results, results_ranges)

    for metric_name, metric_data in data.items():
        # for param_name, values in metric_data.items():
        for param_name in metric_data.keys():
            values = metric_data[param_name]
            # print(values)
            param_values = [value[0] for value in values]
            metric_values = [value[1] for value in values]
            metric_values_ranges = [value[2] for value in values]

            if param_name == consts.MLP.HIDDEN_LAYER_SIZES:
                param_values = [str(len(param_value)) + '*' + str(param_value[0]) for param_value in param_values]
            else:
                param_values = [str(param_value) for param_value in param_values]  # str() even is parameter is a number because numeric parameters are defined in a discrete set.

            graphic_name = output + "_-_" + prediction_function_name + "_-_" + param_name + "_-_" + metric_name
            print_parameters_comparison(param_name, param_values, metric_name, metric_values, metric_values_ranges, graphic_name)


def test_hyper_parameters(prediction_function, features, output, train_iterations_number=10, **parameters_range):
    """
    Test combinations for  all values given for each parameter

    To set a values range for a parameter, one has to give it as "param_name=[val1, val2, ..., valn]".

    :param prediction_function: function that makes the predictions
    :param features: features to train predictors
    :param output: prediction task
    :param train_iterations_number: number of iterations to train model, the final result is the mean of all iterations results
    :param parameters_range: range of values for each parameter
    :return: the best parameters set
    """

    results = []
    results_ranges = []
    best_param_set = None
    param_sets = __initialize_hyper_parameters_sets(**parameters_range)
    print("There are", len(param_sets), "parameters sets and", train_iterations_number, "train iterations.", file=stderr)

    # Initialize progress bar:
    progress_bar = ProgressBar(len(param_sets))
    progress_bar.initialize()

    # Run tests:
    for hyper_parameters in param_sets:
        # prediction_metrics = None
        prediction_metrics_all_values = None

        # Compute a mean value to make results more accurate:
        for it in range(train_iterations_number):
            # print("Iteration ", it, " of ", train_iterations_number, ":", sep="")
            model, it_prediction_metrics = prediction_function(
                features,
                output,
                False, False, False,
                **hyper_parameters
            )
            # if prediction_metrics is None:  # prediction_metrics_all_values is None too
            if prediction_metrics_all_values is None:
                # prediction_metrics = {**it_prediction_metrics}
                prediction_metrics_all_values = {k: [v] for k, v in it_prediction_metrics.items()}
            else:
                for metric_name, it_metric_value in it_prediction_metrics.items():
                    # prediction_metrics[metric_name] += it_metric_value
                    prediction_metrics_all_values[metric_name].append(it_metric_value)

        # for metric_name, metric_value in prediction_metrics.items():
        #     prediction_metrics[metric_name] = metric_value / train_iterations_number
        prediction_metrics = {metric_name: sum(metric_values) / train_iterations_number for metric_name, metric_values in prediction_metrics_all_values.items()}
        results = [*results, {**hyper_parameters, **prediction_metrics}]  # If there have been an error in model training, it will be shown in CSV file by a NAN value for the metric.
        results_ranges = [*results_ranges, {**hyper_parameters, **prediction_metrics_all_values}]

        if best_param_set is None:
            best_param_set = [(metric_name, hyper_parameters, metric_value) if metric_value is not None else (metric_name, None, None) for metric_name, metric_value in prediction_metrics.items()]
        else:
            for i in range(len(best_param_set)):
                metric_name, best_hyper_parameters, metric_value = best_param_set[i]
                if isnan(prediction_metrics[metric_name]):  # If True, the parameters caused an error ...
                    continue  # ... so they can't be the best parameter set.
                if None in [best_hyper_parameters, metric_value] or isnan(metric_value):  # If True, the best parameters haven't been set for this metric ...
                    continue  # ... so they have to be now.
                if abs(abs(consts.PREDICTION_METRICS_OPTIMAL_VALUES[metric_name]) - abs(prediction_metrics[metric_name])) < abs(abs(consts.PREDICTION_METRICS_OPTIMAL_VALUES[metric_name]) - abs(metric_value)):
                    best_param_set[i] = (metric_name, hyper_parameters, prediction_metrics[metric_name])

        # Update progress bar:
        progress_bar.update()

    # End progress bar:
    progress_bar.finalize()

    headers = [*parameters_range.keys(), *[bps[0] for bps in best_param_set]]  # Ordered
    ordered_results = [[result[key] for key in headers] for result in results]

    write_csv(join(get_csv_folder_path(), output[0] + "_-_" + prediction_function.__name__ + consts.CSV), [headers, *ordered_results])
    graphic_results = []
    for r in results:
        if sum([int(True if isnan(r[key]) else False) for key in [bps[0] for bps in best_param_set]]) > 0:
            # print("Excluded parameters set:", r, file=stderr)
            continue  # NAN values aren't taken into account in the graphic to keep accuracy of right values.
        graphic_results.append(r)
    print_parameters_comparisons(prediction_function.__name__, best_param_set, graphic_results, results_ranges, *output)

    return best_param_set


def export_best_params_set(output, prediction_function, best_param_set):
    """
    Create a CSV file containing best parameters sets for each prediction tasks and prediction functions.

    :param output: prediction task
    :param prediction_function: prediction technique
    :param best_param_set: best parameters
    """

    param_names = [param_name for param_name in best_param_set[0][1].keys()]
    headers = ['output', 'prediction_function', 'param_names', 'metric', 'metric_value']
    ordered_results = [[output, prediction_function.__name__, ", ".join([str(param_name) + '=' + str(bps[1][param_name]) for param_name in param_names]) , bps[0], bps[2]] for bps in best_param_set]

    file_path = join(get_csv_folder_path(), consts.BEST_PARAM_SET)
    if not isfile(file_path):
        write_csv(file_path, [headers])
    write_csv(file_path, ordered_results, append=True)


def compare_hyper_parameters(features, *tasks):
    """
    Compare all hyper parameters combinations for all predictors

    :param features: features to train predictors
    """

    train_iterations_number = 10

    max_iter = {  # Max number of iterations for all predictors
        # 10_000,
        # 100_000,
        # 1_000_000,
        10_000_000,
        # -1,  # Warning: Without limitation, some kernel don't converge.
    }

    linear_params_ranges = {
        consts.LinearRegression.FIT_INTERCEPT: [True, False],
        consts.LinearRegression.NORMALIZE: [True, False],
        consts.LinearRegression.COPY_X: [True, False],
        # consts.LinearRegression.N_JOBS: [-1],
        consts.LinearRegression.POSITIVE: [True, False],
    }

    # layer_sizes = [n for n in range(10, 301, 50)]
    # layer_sizes = [n for n in range(1, 20, 4)]
    layer_sizes = [1, 2, 3, *[n for n in range(4, 20, 4)], *[n for n in range(50, 301, 50)]]
    # layer_sizes = [1, 2, 3, *[n for n in range(4, 20, 4)]]
    # layers_numbers = [n for n in range(10, 101, 50)]
    # layers_numbers = [n for n in range(1, 10, 2)]
    layers_numbers = [1, 2, *[n for n in range(3, 10, 2)], 50, 100]
    # layers_numbers = [1, 2, *[n for n in range(3, 10, 2)]]
    layers = []
    for layer_size in layer_sizes:
        layers.extend([tuple(layer_size for _ in range(layers_number)) for layers_number in layers_numbers])
    # print(len(layers))
    # layers = [(n, ) for n in range(10, 101, 20)]  # TODO: Only for tests ...

    mlp_params_ranges = {
        consts.MLP.HIDDEN_LAYER_SIZES: layers,
        consts.MLP.ACTIVATION: [
            # consts.MLP.IDENTITY,
            # consts.MLP.LOGISTIC,
            consts.MLP.TANH,
            consts.MLP.RELU,  # Default Value
        ],
        consts.MLP.SOLVER: [
            consts.MLP.LBFGS,
            consts.MLP.SGD,
            consts.MLP.ADAM,  # Default Value
        ],
        # consts.MLP.ALPHA: [0.0001],
        # consts.MLP.BATCH_SIZE: [consts.MLP.AUTO],
        # consts.MLP.LEARNING_RATE: [
        #     consts.MLP.CONSTANT,
        #     consts.MLP.INVSCALING,
        #     consts.MLP.ADAPTIVE,
        # ],
        # consts.MLP.LEARNING_RATE_INIT: [0.001],
        # consts.MLP.POWER_T: [0.5],
        # consts.MLP.MAX_ITER: max_iter,
        # consts.MLP.SHUFFLE: [True],
        # consts.MLP.RANDOM_STATE: [None],
        # consts.MLP.TOL: [0.0001],
        # consts.MLP.VERBOSE: [False],
        # consts.MLP.WARM_START: [False],
        # consts.MLP.MOMENTUM: [0.9],
        # consts.MLP.NESTEROVS_MOMENTUM: [True],
        consts.MLP.EARLY_STOPPING: [
            True,
            # False
        ],
        # consts.MLP.VALIDATION_FRACTION: [0.1],
        # consts.MLP.BETA_1: [0.9],
        # consts.MLP.BETA_2: [0.999],
        # consts.MLP.EPSILON: [1e-08],
        # consts.MLP.N_ITER_NO_CHANGE: [10],
        # consts.MLP.MAX_FUN: [15_000],
    }

    svm_main_params_ranges = {
        consts.SVM.KERNEL: [
            consts.PREDICTION_KERNEL_LINEAR,
            consts.PREDICTION_KERNEL_POLY,
            consts.PREDICTION_KERNEL_RBF,  # Default value
            consts.PREDICTION_KERNEL_SIGMOID,
        ],
        consts.SVM.GAMMA: [  # Not for linear kernel
            consts.SVM.GAMMA_SCALE,  # Default value
            consts.SVM.GAMMA_AUTO,
        ],
        consts.SVM.MAX_ITER: max_iter,
        consts.SVM.TOL: [
            1,
            1e-1,
            1e-3,  # Default value
            1e-6,
            1e-9,
            1e-12,
        ],
        consts.SVM.SHRINKING: [
            True,  # Default value
            False,
        ],
    }

    svc_params_ranges = {
        **svm_main_params_ranges,
        # consts.SVM.PROBABILITY: [
        #
        # ],
        consts.SVM.DECISION_FUNCTION_SHAPE: [
            consts.SVM.DECISION_FUNCTION_SHAPE_OVO,
            consts.SVM.DECISION_FUNCTION_SHAPE_OVR,  # Default value
        ],
    }

    random_forest_params_ranges = {
        "n_estimators": [
            n for n in range(10, 101, 10)
        ],
        "max_depth": [
            n for n in range(10, 101, 10)
        ],
        "min_samples_split": [
            n for n in range(2, 11)
        ],
        "min_samples_leaf": [
            n for n in range(1, 11)
        ],
    }

    svr_params_ranges = {
        **svm_main_params_ranges,
    }

    regression_functions = {
        perform_linear_regression: linear_params_ranges,
        perform_mlp_regression: mlp_params_ranges,
        perform_svr_regression: svr_params_ranges,
    }

    classification_functions = {
        perform_svc_classification: svc_params_ranges,
        perform_random_forest_classification: random_forest_params_ranges,
    }
    outputs = {
        consts.OUTPUT_IS_SINGLE_SOLUTION: classification_functions,
        consts.OUTPUT_NB_SOLUTIONS: regression_functions,
        consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES: classification_functions,
        consts.OUTPUT_NB_SOLUTION_CLASSES: regression_functions,
    }

    # Reset best parameters export:
    best_param_sets_path = join(get_csv_folder_path(), consts.BEST_PARAM_SET)
    if exists(best_param_sets_path):
        os.remove(best_param_sets_path)

    running_times = {}
    hyper_parameters_start_time = time()

    for output, prediction_functions in outputs.items():
        if len(tasks) > 0 and output not in tasks:
            continue
        for prediction_function, params_ranges in prediction_functions.items():
            prediction_function_name = str(prediction_function.__name__).replace("_", " ")[8:]
            print("####", output, ":", prediction_function_name, "####")
            prediction_function_start_time = time()
            best_param_set = test_hyper_parameters(prediction_function, features, [output], train_iterations_number=train_iterations_number, **params_ranges)
            prediction_function_end_time = time() - prediction_function_start_time
            export_best_params_set(output, prediction_function, best_param_set)
            export_running_time(output + "_-_" + prediction_function_name, prediction_function_end_time)
            print("Running time for " + prediction_function_name + ":", prediction_function_end_time, "seconds")

            print("Best parameters sets:")
            for (metric_name, metric_best_param_set, metric_value) in best_param_set:
                print("\tMetric:", metric_name, "=", metric_value)
                for param, value in metric_best_param_set.items():
                    print("\t\t", param, ": ", value, sep="")
            print()

    hyper_parameters_end_time = time() - hyper_parameters_start_time
    export_running_time("all hyper-parameters", hyper_parameters_end_time)
    print("Running time for hyper-parameters tuning:", hyper_parameters_end_time, "seconds")
