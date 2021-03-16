'''
Created on Feb 25, 2021

@author: Laurent PEREIRA DA SILVA
'''
import os
import consts
import path
import util
from deprecated import deprecated

import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def generate_plot(x_values, y_values, graphic_title):
    """This method generate a plot using matplotlib.pyplot

    :param x_values: a list of values used for the Y Axis
    :type x_values: integer list
    :param y_values: a list of values used for the Y Axis
    :type y_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    """
    graphic_title = str(graphic_title)
    graphic_title = util.prediction_name_refactor(graphic_title)
    filename = graphic_title+"_plot.png"
    path_to_file = os.path.join(path.get_graphics_folder_path(), filename)

    print("Generating plot for "+graphic_title, "\n")
    plt.plot(x_values, y_values, marker='o', c='blue', markersize=5)
    plt.title(graphic_title)
    plt.savefig(path_to_file)
    plt.close()


@deprecated("This function is deprecated, use 'generate_boxplot_clean()' instead")
def generate_boxplot(outputs_values, predicted_values, graphic_title):
    """ Deprecated method, use generate_boxplot_clean() instead
    This method generate a boxplot using matplotlib.pyplot

    :param outputs_values: a list of values used for the Y Axis
    :type outputs_values: integer list
    :param predicted_values: a list of values used for the Y Axis
    :type predicted_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    """
    """
    # some tests (delete once the method is complete)
    print(outputs_values)
    print(type(outputs_values))
    print(outputs_values.shape)
    print(predicted_values)
    print(type(predicted_values))
    print(predicted_values.shape) # the shape are not the same, so I need to add something for it to work (see below)

    # Variables Initialisation
    outputs_values_updated = []  # will contain all outputs values
    predicted_values_updated = []  # will contain all predicted values
    data = []  # will contain multiples dataset, each one stands fora different boxplot
    outputs_dictionary = {}
    predicted_dictionary = {}
    x_axis_names = []

    # Initial setup for file saving
    graphic_title = str(graphic_title)
    graphic_title = util.prediction_name_refactor(graphic_title)
    filename = graphic_title + "_boxplot.png"
    path_to_file = os.path.join(path.get_graphics_folder_path(), filename)

    # transforming dataset to list with same shape
    for x in np.nditer(
            outputs_values):  # method to select all values inside the ndarray : https://numpy.org/doc/stable/reference/arrays.nditer.html
        outputs_values_updated.append(float(x))
        # print(outputs_values_updated)

    for y in np.nditer(predicted_values):
        # predicted_values_updated.append(float("{:.3f}".format(y))) # https://stackoverflow.com/a/455634
        predicted_values_updated.append(float(y))
        # print(predicted_values_updated)

    # Pre-processing
    max_output = max(outputs_values_updated)
    print("max output : " + str(max_output))
    max_predict = max(predicted_values_updated)
    print("max predict : " + str(max_predict))

    # adding all index of outputs into a dictionary
    print("Adding all outputs index")
    for i in range(1, int(max_output), 10):  # loop from 1 to the max value of the list, with a step of 10
        # print(i)
        tmp = i + 9
        tmp_string = str(i) + ":" + str(tmp)
        outputs_dictionary[tmp_string] = 0
        x_axis_names.append(tmp_string)
        print(x_axis_names)
        # print(outputs_dictionary)
        tmp_list = []
        for x in outputs_values_updated:
            if i <= x < tmp + 1:
                print(x)
                # return the index of this value
                print(outputs_values_updated.index(
                    x))  # If multiple occurrences of the same value, return only the first index
                tmp_list.append(outputs_values_updated.index(x))
                print(tmp_list)
        outputs_dictionary[tmp_string] = tmp_list
        print(outputs_dictionary)

    # adding all index of predicted values into a dictionary (maybe useless)
    # print("Adding all predicted values index")
    for i in range(1, int(max_predict), 10):  # loop from 1 to the max value of the list, with a step of 10
        # print(i)
        tmp = i + 9
        tmp_string = str(i) + ":" + str(tmp)
        predicted_dictionary[tmp_string] = 0
        # print(predicted_dictionary)
        tmp_list = []
        for x in predicted_values_updated:
            if i <= x < tmp + 1:
                # print(x)
                # return the index of this value
                # print(predicted_values_updated.index(x))  # If multiple occurrences of the same value, return only the first index
                tmp_list.append(predicted_values_updated.index(x))
                # print(tmp_list)
        predicted_dictionary[tmp_string] = tmp_list
        # print(predicted_dictionary)

    # collecting predicted values at the corresponding indexes
    print("Collecting predicted values at corresponding indexes")
    for i in range(1, int(max_output), 10):  # loop from 1 to the max value of the list, with a step of 10
        tmp = i + 9
        tmp_string = str(i) + ":" + str(tmp)
        print(outputs_dictionary[tmp_string])
        tmp_index_list = outputs_dictionary[tmp_string]
        tmp_list = []
        if len(tmp_index_list) == 0:
            data.append([])
            print(data)
        else:
            for x in tmp_index_list:
                tmp_list.append(float(predicted_values_updated[x]))
                print(tmp_list)
            data.append(tmp_list)
            print(data)

    # Generating boxplot
    print("Generating boxplot for " + graphic_title, "\n")
    axes = plt.gca()
    plt.boxplot(data)
    plt.title(graphic_title)
    axes.set_xticklabels(x_axis_names)
    plt.savefig(path_to_file)
    plt.close()
    """

    return generate_boxplot_clean(outputs_values, predicted_values, graphic_title)


def generate_boxplot_clean(outputs_values, predicted_values, graphic_title):
    """This method generate a boxplot using matplotlib.pyplot

    :param outputs_values: a list of values used for the Y Axis
    :type outputs_values: integer list
    :param predicted_values: a list of values used for the Y Axis
    :type predicted_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    """

    # Variables Initialisation
    outputs_values_updated = []  # will contain all outputs values
    predicted_values_updated = []  # will contain all predicted values
    data = []  # will contain multiples dataset, each one stands fora different boxplot
    outputs_dictionary = {}  # will contain position of elements from outputs_values_updated
    x_axis_names = []  # will contain labels for the x axis
    interval_value = 10  # Value used to set the interval range (initial value : 10)

    # Initial setup for file saving
    graphic_title = str(graphic_title)
    graphic_title = util.prediction_name_refactor(graphic_title)
    filename = graphic_title+"_boxplot.png"
    path_to_file = os.path.join(path.get_graphics_folder_path(), filename)

    # transforming dataset to list with same shape
    for x in np.nditer(outputs_values):  # method to select all values inside the ndarray : https://numpy.org/doc/stable/reference/arrays.nditer.html
        outputs_values_updated.append(float(x))

    for y in np.nditer(predicted_values):
        predicted_values_updated.append(float(y))

    # collecting max value of both lists
    max_output = max(outputs_values_updated)

    # adding all index of outputs into a dictionary
    for i in range(1, int(max_output), interval_value):  # loop from 1 to the max value of the list, with a step of 10
        tmp = i + (interval_value - 1)
        tmp_string = str(i) + ":" + str(tmp)
        outputs_dictionary[tmp_string] = 0  # adding an element to the dict with the key and an initial value
        x_axis_names.append(tmp_string)
        tmp_list = []  # temporary list that will contain the indexes of elements within the actual range
        for x in outputs_values_updated:
            if i <= x < tmp+1:
                tmp_list.append(outputs_values_updated.index(x))  # If multiple occurrences of the same value, only the first index will be added
        outputs_dictionary[tmp_string] = tmp_list

    # collecting predicted values at the corresponding indexes
    for i in range(1, int(max_output), interval_value):  # loop from 1 to the max value of the list, with a step of 10
        tmp = i + (interval_value - 1)
        tmp_string = str(i) + ":" + str(tmp)
        tmp_index_list = outputs_dictionary[tmp_string]
        tmp_list = []
        if len(tmp_index_list) == 0:
            data.append([])
        else:
            for x in tmp_index_list:
                # tmp_list.append(float(predicted_values_updated[x]))
                tmp_list.append(float(predicted_values_updated[x]) - float(outputs_values_updated[x]))
            data.append(tmp_list)

    # Generating boxplot
    print("Generating boxplot for "+graphic_title, "\n")
    axes = plt.gca()
    plt.boxplot(data)
    plt.title(graphic_title)
    axes.set_xticklabels(x_axis_names)
    plt.savefig(path_to_file)
    plt.close()
