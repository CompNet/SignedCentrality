#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to hyper parameters tes and comparison.

@author: Virgile Sucal
"""

from sys import stderr
from deprecated import deprecated
from sklearn import metrics
import consts
from collect.collect_predicted_values import collect_predicted_values
from prediction import initialize_hyper_parameters, initialize_data, process_graphics
from util import write_csv


def pop(dictionary):
    key = dictionary.keys()[0]
    value = dictionary[key]
    del dictionary[key]
    return key, value


def __define_output_file_name(hyper_parameters, prediction_function=None, ext=None):
    return prediction_function + "_-_" if prediction_function is not None else "" + ";".join([str(key) + "=" + str(value) for key, value in hyper_parameters.items()]) + ext if ext is not None else ""


def __initialize_all_models(results=None, param_pool=None, **kwargs):
    if results is None:
        results = []
    if param_pool is None:
        param_pool = {}

    if len(kwargs) == 0:
        print(__define_output_file_name(param_pool))
        return [*results, {**param_pool}]

    param, values = pop(kwargs)

    for value in values:
        results = [*results, *__initialize_all_models(results, {**param_pool, param: value}, **kwargs)]

    return results


def test_hyper_parameters(prediction_function, features, export_path, **parameters_range):
    """
    Test combinations for  all values given for each parameter

    To set a values range for a parameter, one has to give it as "param_name=[val1, val2, ..., valn]".
    """

    results = []

    for hyper_parameters in __initialize_all_models(**parameters_range):
        model, prediction_metrics = prediction_function(features, __define_output_file_name(hyper_parameters, prediction_function), False, False, True, **hyper_parameters)
        results.append({**parameters_range, **prediction_metrics})

    headers = [*parameters_range.keys(), *results[0][1].keys()]  # Ordered
    ordered_results = [[result[key] for key in headers] for result in results]

    write_csv(export_path, [headers, *ordered_results])


def compare_hyper_parameters(features, export_path):

    linear_params_ranges = {
        consts.LinearRegression.FIT_INTERCEPT: [True, False],
        consts.LinearRegression.NORMALIZE: [True, False],
        consts.LinearRegression.COPY_X: [True, False],
        consts.LinearRegression.N_JOBS: [-1],
        consts.LinearRegression.POSITIVE: [True, False],
    }

    mlp_params_ranges = {
        consts.MLP.HIDDEN_LAYER_SIZES: [
            (
                layer_size for layer_size in range(10, 301, 10)
            ) for layer_number in range(10, 101, 10)
        ],
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
        consts.MLP.ALPHA: [0.0001],
        consts.MLP.BATCH_SIZE: [consts.MLP.AUTO],
        consts.MLP.LEARNING_RATE: [
            consts.MLP.CONSTANT,
            consts.MLP.INVSCALING,
            consts.MLP.ADAPTIVE,
        ],
        consts.MLP.LEARNING_RATE_INIT: [0.001],
        consts.MLP.POWER_T: [0.5],
        consts.MLP.MAX_ITER: [
            n for n in range(200, 1_001, 100)
        ],
        consts.MLP.SHUFFLE: [True],
        consts.MLP.RANDOM_STATE: [None],
        consts.MLP.TOL: [0.0001],
        consts.MLP.VERBOSE: [False],
        consts.MLP.WARM_START: [False],
        consts.MLP.MOMENTUM: [0.9],
        consts.MLP.NESTEROVS_MOMENTUM: [True],
        consts.MLP.EARLY_STOPPING: [
            True,
            False
        ],
        consts.MLP.VALIDATION_FRACTION: [0.1],
        consts.MLP.BETA_1: [0.9],
        consts.MLP.BETA_2: [0.999],
        consts.MLP.EPSILON: [1e-08],
        consts.MLP.N_ITER_NO_CHANGE: [10],
        consts.MLP.MAX_FUN: [15_000],
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
        consts.SVM.MAX_ITER: {
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            # -1,  # Warning: Without limitation, some kernel don't converge.
        },
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
