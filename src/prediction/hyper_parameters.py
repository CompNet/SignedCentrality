#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to hyper parameters tes and comparison.

@author: Virgile Sucal
"""
from math import isnan
from os.path import abspath, dirname, join
from sys import stderr
from deprecated import deprecated
from sklearn import metrics
import consts
import sys
from collect.collect_graphics import generate_plot, generate_errorbar_plot
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics
from prediction.regression import perform_linear_regression, perform_mlp_regression, perform_svr_regression
from util import write_csv, ProgressBar
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


def __initialize_graphic_data(best_param_set, results):
    """
    Initialise data for graphic functions

    :param best_param_set: best parameters set
    :param results: all parameters set
    :return: the data
    """

    data = {}

    for (metric_name, param_set, metric_value) in best_param_set:
        metric_data = {param_name: [(param_set[param_name], metric_value)] for param_name in param_set.keys()}

        for result in results:
            non_optimal_values = []
            for param, value in param_set.items():  # Don't use result, because it contains more columns than best_param_set.
                if value != result[param]:
                    non_optimal_values.append(param)
            if len(non_optimal_values) != 1:
                continue

            tested_param_name = non_optimal_values[0]
            metric_data[tested_param_name].append((result[tested_param_name], result[metric_name]))

        data[metric_name] = {param_name: tuple(data_list) for param_name, data_list in metric_data.items()}

    return data


def print_parameters_comparison(param_name, param_values, metric_name, metric_values, graphic_name):
    """
    Export graphic for all values of all hyper parameters

    :param param_name: name of the parameter
    :param param_values: values for the parameter
    :param metric_name: name of the metric
    :param metric_values: values for the metric
    :param graphic_name: name of the graphic
    """

    param_name = str(param_name)
    metric_name = str(metric_name)
    generate_errorbar_plot(param_values, metric_values, graphic_name, param_name.replace("_", " "), metric_name.replace("_", " "), print_title=False, dash_between_name_and_plot=True)


def print_parameters_comparisons(prediction_function_name, best_param_set, results, output):
    """
    Export graphic for all values of all hyper parameters

    :param prediction_function_name: name of the function that makes the predictions
    :param best_param_set: best parameters set
    :param results: all parameters set
    :param output: prediction task
    """

    data = __initialize_graphic_data(best_param_set, results)

    for metric_name, metric_data in data.items():
        for param_name, values in metric_data.items():
            param_values = [value[0] for value in values]
            metric_values = [value[1] for value in values]

            if param_name == consts.MLP.HIDDEN_LAYER_SIZES:
                param_values = [str(len(param_value)) + '*' + str(param_value[0]) for param_value in param_values]
            else:
                param_values = [str(param_value) for param_value in param_values]  # str() even is parameter is a number because numeric parameters are defined in a discrete set.

            graphic_name = output + "_-_" + prediction_function_name + "_-_" + param_name + "_-_" + metric_name
            print_parameters_comparison(param_name, param_values, metric_name, metric_values, graphic_name)


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
    best_param_set = None
    param_sets = __initialize_hyper_parameters_sets(**parameters_range)
    print("There are", len(param_sets), "parameters sets and", train_iterations_number, "train iterations.", file=stderr)

    # Initialize progress bar:
    progress_bar = ProgressBar(len(param_sets))
    progress_bar.initialize()

    # Run tests:
    for hyper_parameters in param_sets:
        prediction_metrics = None

        # Compute a mean value to make results more accurate:
        for it in range(train_iterations_number):
            # print("Iteration ", it, " of ", train_iterations_number, ":", sep="")
            model, it_prediction_metrics = prediction_function(
                features,
                output,
                False, False, False,
                **hyper_parameters
            )
            if prediction_metrics is None:
                prediction_metrics = {**it_prediction_metrics}
            else:
                for metric_name, it_metric_value in it_prediction_metrics.items():
                    prediction_metrics[metric_name] += it_metric_value

        for metric_name, metric_value in prediction_metrics.items():
            prediction_metrics[metric_name] = metric_value / train_iterations_number
        results = [*results, {**hyper_parameters, **prediction_metrics}]  # If there have been an error in model training, it will be shown in CSV file by a NAN value for the metric.

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
    print_parameters_comparisons(prediction_function.__name__, best_param_set, graphic_results, *output)

    return best_param_set


def compare_hyper_parameters(features):
    """
    Compare all hyper parameters combinations for all predictors

    :param features: features to train predictors
    """

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

    layer_sizes = [n for n in range(10, 301, 50)]
    layers_numbers = [n for n in range(10, 101, 50)]
    layers = []
    for layer_size in layer_sizes:
        layers.extend([tuple(layer_size for _ in range(layers_number)) for layers_number in layers_numbers])
    print(len(layers))

    mlp_params_ranges = {
        consts.MLP.HIDDEN_LAYER_SIZES: layers,
        consts.MLP.ACTIVATION: [
            consts.MLP.IDENTITY,
            consts.MLP.LOGISTIC,
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
        # consts.MLP.EARLY_STOPPING: [
        #     True,
        #     False
        # ],
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
        consts.SVM.PROBABILITY: [

        ],
        consts.SVM.DECISION_FUNCTION_SHAPE: [
            consts.SVM.DECISION_FUNCTION_SHAPE_OVO,
            consts.SVM.DECISION_FUNCTION_SHAPE_OVR,  # Default value
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
        # perform_svc_classification: svc_params_ranges,
    }

    outputs = {
        # consts.OUTPUT_IS_SINGLE_SOLUTION: classification_functions,
        consts.OUTPUT_NB_SOLUTIONS: regression_functions,
        # consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES: classification_functions,
        consts.OUTPUT_NB_SOLUTION_CLASSES: regression_functions,
    }

    for output, prediction_functions in outputs.items():
        for prediction_function, params_ranges in prediction_functions.items():
            print("####", output, ":", prediction_function.__name__, "####")
            best_param_set = test_hyper_parameters(prediction_function, features, [output], **params_ranges)
            print("Best parameters sets:")
            for (metric_name, metric_best_param_set, metric_value) in best_param_set:
                print("\tMetric:", metric_name, "=", metric_value)
                for param, value in metric_best_param_set.items():
                    print("\t\t", param, ": ", value, sep="")
            print()
