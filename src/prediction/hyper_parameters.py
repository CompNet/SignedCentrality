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

