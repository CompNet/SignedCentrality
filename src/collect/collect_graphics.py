'''
Created on Feb 25, 2021

@author: Laurent PEREIRA DA SILVA
'''
import os
import consts
import path

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
    graphic_title = str(graphic_title)
    filename = graphic_title+"_boxplot.png"
    path_to_file = os.path.join(path.get_graphics_folder_path(), filename)

    print("Generating boxplot for "+graphic_title)
    plt.boxplot(x_values, y_values) # issue here
    plt.title(graphic_title)
    plt.savefig(path_to_file)
    plt.close()
