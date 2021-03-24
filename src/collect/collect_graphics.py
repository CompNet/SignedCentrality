'''
Created on Feb 25, 2021

@author: Laurent PEREIRA DA SILVA
'''
import os
from statistics import stdev
import consts
import path
import numpy
import matplotlib.pyplot as plt


def __make_file_path(graphic_title, plot_type: str = None, add_plot_to_name=True, dash_between_name_and_plot=False):
    """
    Creates a path for a graphic

    :param graphic_title: the title of the graphic
    :param add_plot_to_name: True if plot type must be added
    :param dash_between_name_and_plot: True if plot type must be preceded by a dash
    :return: the computed path
    """

    return os.path.join(path.get_graphics_folder_path(), str(graphic_title + str(str(
        str("_-" if dash_between_name_and_plot else "") + "_" + str(
            plot_type)) if plot_type is not None and add_plot_to_name else "") + consts.PNG))


def __make_plot(plot_function, graphic_title, x_label=None, y_label=None, print_title=True, add_plot_to_name=True, dash_between_name_and_plot=False, verbose=False, *args, **kwargs):
    """
    Create a generic plot

    :param plot_function: function to create the plot
    :param graphic_title: the title of the graphic
    :param add_plot_to_name: True if plot type must be added
    :param dash_between_name_and_plot: True if plot type must be preceded by a dash
    :param verbose: True if information must be printed
    :param kwargs: parameters for plot_function
    """

    graphic_title = str(graphic_title)
    path_to_file = __make_file_path(graphic_title, plot_function.__name__, add_plot_to_name, dash_between_name_and_plot)

    if verbose:
        print("Generating " + plot_function.__name__ + " for " + graphic_title)
    plot_function(*args, **kwargs)
    if print_title:
        plt.title(graphic_title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(path_to_file)
    plt.close()


def generate_plot(x_values, y_values, graphic_title, x_label=None, y_label=None, print_title=True, add_plot_to_name=True,dash_between_name_and_plot=False, verbose=False):
    """
    This method generate a plot using matplotlib.pyplot

    :param x_values: a list of values used for x axis
    :type x_values: integer list
    :param y_values: a list of values used for y axis
    :type y_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    :param x_label: label for x axis
    :param y_label: label for y axis
    :param add_plot_to_name: True if plot type must be added
    :param dash_between_name_and_plot: True if plot type must be preceded by a dash
    :param verbose: True if information must be printed
    """

    __make_plot(
        plt.plot,
        graphic_title,
        x_label, y_label,
        print_title,
        add_plot_to_name, dash_between_name_and_plot,
        verbose,
        x_values, y_values,
        marker='o', c='blue', markersize=5
    )


def generate_errorbar_plot(x_values, y_values, graphic_title, x_label=None, y_label=None, print_title=True, add_plot_to_name=True,dash_between_name_and_plot=False, verbose=False):
    """
    This method generate an errorbar plot using matplotlib.pyplot

    :param x_values: a list of values used for x axis
    :type x_values: integer list
    :param y_values: a list of values used for y axis
    :type y_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    :param x_label: label for x axis
    :param y_label: label for y axis
    :param add_plot_to_name: True if plot type must be added
    :param dash_between_name_and_plot: True if plot type must be preceded by a dash
    :param verbose: True if information must be printed
    """

    stddev = None
    if len(y_values) > 1:
        stddev = stdev([float(y) for y in y_values])
    __make_plot(
        plt.errorbar,
        graphic_title,
        x_label, y_label,
        print_title,
        add_plot_to_name, dash_between_name_and_plot,
        verbose,
        x_values, y_values,
        yerr=stddev, linestyle = 'none', marker='o', c='blue', markersize=5
    )


def generate_boxplot(x_values, y_values, graphic_title, add_plot_to_name=True, dash_between_name_and_plot=False,
                     verbose=False):
    """This method generate a boxplot using matplotlib.pyplot

    :param x_values: a list of values used for the Y Axis
    :type x_values: integer list
    :param y_values: a list of values used for the Y Axis
    :type y_values: integer list
    :param graphic_title: the title of the graphic
    :type graphic_title: string
    :param add_plot_to_name: True if plot type must be added
    :param dash_between_name_and_plot: True if plot type must be preceded by a dash
    :param verbose: True if information must be printed
    """

    print(type(x_values))
    print(x_values.shape)
    print(type(y_values))
    print(y_values.shape)  # the shape are not the same, so I need to add something for it to work (see below)

    graphic_title = str(graphic_title)
    path_to_file = __make_file_path(graphic_title, "boxplot", add_plot_to_name, dash_between_name_and_plot)

    if verbose:
        print("Generating boxplot for " + graphic_title)
    # plt.boxplot(x_values, y_values)  # issue here
    # plt.boxplot(x_values.any(), y_values.any()) # issue here
    plt.boxplot(x_values, y_values.all())  # only works if y_values is the predicted dataset
    plt.title(graphic_title)
    plt.savefig(path_to_file)
    plt.close()
