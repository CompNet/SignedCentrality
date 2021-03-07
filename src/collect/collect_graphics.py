'''
Created on Feb 25, 2021

@author: Laurent PEREIRA DA SILVA
'''
import os
import consts
import path
import util

import numpy
import matplotlib.pyplot as plt


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

    print("Generating plot for "+graphic_title)
    plt.plot(x_values, y_values, marker='o', c='blue', markersize=5)
    plt.title(graphic_title)
    plt.savefig(path_to_file)
    plt.close()


def generate_boxplot(x_values, y_values, graphic_title):
    """This method generate a boxplot using matplotlib.pyplot

    :param x_values: a list of values used for the Y Axis
    :type x_values: integer list
    :param y_values: a list of values used for the Y Axis
    :type y_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    """

    print(type(x_values))
    print(x_values.shape)
    print(type(y_values))
    print(y_values.shape) # the shape are not the same, so I need to add something for it to work (see below)
    graphic_title = str(graphic_title)
    graphic_title = util.prediction_name_refactor(graphic_title)
    filename = graphic_title+"_boxplot.png"
    path_to_file = os.path.join(path.get_graphics_folder_path(), filename)

    print("Generating boxplot for "+graphic_title)
    # plt.boxplot(x_values, y_values)  # issue here
    # plt.boxplot(x_values.any(), y_values.any()) # issue here
    plt.boxplot(x_values, y_values.all()) # only works if y_values is the predicted dataset
    plt.title(graphic_title)
    plt.savefig(path_to_file)
    plt.close()
