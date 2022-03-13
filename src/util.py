'''
Created on Sep 23, 2020

@author: nejat
@author: Virgile Sucal
'''

import csv
import itertools as iter
from os.path import join, isfile
import numpy as np
import math
import consts
from csv import reader, writer, Sniffer, Dialect
from sys import float_info, stdout
from igraph import Graph
import path


def format_2digits(x):
    """This method formats the given number (either integer or float)
        by forcing to have 4 digits.
       
    :param x: number
    :type x: int or float
    """
    return ('%.2f' % x)
  
  
def format_4digits(x):
    """This method formats the given number (either integer or float)
        by forcing to have 4 digits.
       
    :param x: number
    :type x: int or float
    """
    return ('%.4f' % x)


def which(values):
    """This program finds the indexes of the elements that are True.
    The type of 'values' has to be boolean.
       
    values: boolean values
    """
    for value in values:
        if not isinstance(value, (bool, np.bool_)):
            raise Exception('which() function expects only boolean values')
    indxs = [indx for indx, bool_elt in enumerate(values) if bool_elt]
    return indxs




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
  
  
def generate_uniform_membership(n, l0):
    """This method creates a membership data, where the size of the modules
    as equal as possible, i.e. uniform distribution.
       
    :param n: a list of number of nodes
    :type n: int
    :param l0: number of modules from which the underlying graph is created
    :type l0: int
    :param d: density
    :type d: float
    :param prop_mispl: proportion of misplaced links
    :type prop_mispl: float
    """
    membership = []
    nk = math.floor(n / l0)
    module_sizes = [nk] * l0
    nb_remaining = n - (nk * l0)
    for i in range(nb_remaining):
        module_sizes[i] += 1
    for i in range(l0):
        membership.extend([i + 1] * module_sizes[i])
    return membership


def compute_prop_neg(n, l0):
    """This program computes the proportion of negative links
       of a given signed graph, whose the density is 1 
       (i.e. complete signed graph)
       
       n: number of nodes
       l0: number of modules
       membership: membership vector as python tuple
    """
    prop_neg = 0

    membership = generate_uniform_membership(n, l0)
    module_sizes = [membership.count(indx) for indx in range(1, l0 + 1)]
    for m1, m2 in iter.combinations(range(1, l0 + 1), 2):
        n1 = module_sizes[m1 - 1]
        n2 = module_sizes[m2 - 1]
        prop_neg += (n1 * n2) / ((n * (n - 1) / 2))

    return prop_neg


def read_graph(path_name, format=None):
    """
    Read a graph from a file.

    It can read some file formats, such as GraphML or CSV.
    XML-like files such as GraphML have to be written in a standard format.
    See example below for GraphML files.

    If format is None, it will be detected automatically.
    It might cause errors. It is preferred that the format has been set.

    The function creates a Graph with the igraph library.

    :param path_name: path of the file
    :type path_name: str
    :param format: format of the file
    :type format: str
    :return: the graph
    :rtype: igraph.Graph

    Here is an example of how the GraphML file has to be written.
    This GraphML file uses the standards of the igraph.graph.write_graphml() method.

    :Example:

    <?xml version="1.0" encoding="UTF-8"?>
    <graphml
            xmlns="http://graphml.graphdrawing.org/xmlns"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
            http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
        <key id="v_id" for="node" attr.name="id" attr.type="string"/>
        <key id="e_weight" for="edge" attr.name="weight" attr.type="double"/>
        <key id="e_id" for="edge" attr.name="id" attr.type="string"/>
        <graph id="network_a" edgedefault="undirected">
            <node id="n0">
            <data key="v_id">1</data>
            </node>
            <node id="n1">
            <data key="v_id">2</data>
            </node>
            <node id="n2">
            <data key="v_id">3</data>
            </node>
            <edge source="n0" target="n1">
            <data key="e_weight">1</data>
            <data key="e_id">e0</data>
            </edge>
            <edge source="n1" target="n2">
            <data key="e_weight">1</data>
            <data key="e_id">e1</data>
            </edge>
            <edge source="n2" target="n3">
            <data key="e_weight">1</data>
            <data key="e_id">e2</data>
            </edge>
        </graph>
    </graphml>

    .. warning: In the GraphML file, the attribute which defines the weights of the edges has to be given the name "weight" with the attribute "attr.name".
    """

    graph = None

    if format is not None and format.lower() == consts.FILE_FORMAT_CSV:
        # The separator in CSV files is the comma.
        graph = Graph.Read_Adjacency(path_name, sep=",", comment_char="#", attribute=consts.EDGE_WEIGHT_ATTR)
    else:
        graph = Graph.Read(path_name, format)

    return graph


def write_graph(graph, path_name):
    """
    Write a GraphML file from a Graph.

    This function is used for tests.

    :param graph: graph to write
    :type graph: igraph.Graph
    :param path_name: path of the GraphML file
    :type path_name: str
    """

    graph.write_graphml(path_name)


def get_matrix(graph):
    """
    Returns the adjacency matrix of the given graph.

    This matrix is an instance of the class scipy.sparse.csr_matrix.
    The default igraph.Matrix class isn't used because it doesn't support arithmetic operations.

    :param graph: the graph to be used to extract the adjacency matrix
    :type graph: igraph.Graph
    :return: the adjacency matrix
    :rtype: scipy.sparse.csr_matrix
    """

    return graph.get_adjacency_sparse(consts.EDGE_WEIGHT_ATTR)  # scipy.sparse.csr_matrix


def get_adj_list(graph):
    """
    Returns the adjacency list of the given graph.

    :param graph: the graph to be used to extract the adjacency list
    :type graph: igraph.Graph
    :return: the adjacency list
    :rtype: list
    """

    matrix = get_matrix(graph).toarray()
    adj_list = []
    pairs = []
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if matrix[i][j] != 0 and i <= j and (i != j or i not in pairs):
                if i == j:
                    pairs.append(i)
                adj_list.append([i, j, matrix[i][j]])

    return adj_list


def get_scale(centrality, fit_sign=False):
    """
    Compute the scale value to scale a centrality

    If fit_sign is True, the sign of the scale is changed when the sum of the centrality is negative.
    It is used to make the first clique positive.

    :param centrality: the centrality which the scale is to be computed
    :type centrality: numpy.ndarray
    :param fit_sign: indicates if the sign must be changed
    :type fit_sign: bool
    :return: the scale
    :rtype: float
    """

    max_value = float_info.min  # Minimal value of a float
    for value in centrality:
        if abs(value) > max_value:  # abs(), because the magnitude must be scaled, not the signed value.
            max_value = abs(value)
    scale = 1 / max_value

    if fit_sign and sum(centrality) < 0:  # Makes the first cluster values positive if they aren't.
        scale *= -1  # Values will be inverted when they will be scaled (more efficient).

    return scale


def scale_centrality(centrality, fit_sign=False):
    """
    Scale the given centrality

    If fit_sign is True, the sign of the scale is changed when the sum of the centrality is negative.
    It is used to make the first clique positive.

    :param centrality: the centrality which the scale is to be computed
    :type centrality: numpy.ndarray
    :param fit_sign: indicates if the sign must be changed
    :type fit_sign: bool
    :return: the scaled centrality
    :rtype: numpy.ndarray
    """

    scale_ = get_scale(centrality, fit_sign)

    if scale_ == 1:  # If the centrality has the right signs and if it doesn't have to be scaled, it can be returned.
        return centrality

    return [value * scale_ for value in centrality]  # Else, return a scaled centrality.


def matrix_to_graph(matrix):
    """
    Creates a graph from a numpy adjacency matrix

    :param matrix: the adjacency matrix
    :type matrix: numpy.ndarray
    :param weight_attr: attribute name to use for weights
    :type weight_attr: str
    :return: the graph
    :rtype: igraph.Graph
    """

    length = len(matrix)
    graph = Graph()
    graph.to_directed()  # If an undirected graph is needed, it will be done later.
    graph.add_vertices(length)

    for row in range(length):
        for col in range(length):
            weight = matrix[row, col]
            if weight > 0:
                graph.add_edge(row, col, sign=+1, weight=weight)
            elif weight < 0:
                graph.add_edge(row, col, sign=-1, weight=weight)
            else:
                graph.add_edge(row, col)

    return graph


def read_csv(path: str, remove_headers=True):
    """
    Create a ndarray matrix from a CSV file

    The matrix isn't necessarily the adjacency matrix of a graph.
    It can contain all kinds of data.

    :param path: path of the CSV file
    :type path: str
    :param remove_headers: True if headers must be removed
    :type remove_headers: bool
    :return: the matrix containing data
    :rtype: list of lists
    """

    matrix = None
    csv = []

    with open(path, 'r') as file:
        dialect = Sniffer().sniff(file.read(1024))
        file.seek(0)

        header = Sniffer().has_header(file.read(1024))
        file.seek(0)

        for row in reader(file, dialect):
            csv.append(row)

        matrix = [[csv[i][j] for j in range(int(header and remove_headers), len(csv[i]))] for i in range(int(header and remove_headers), len(csv))]  # int(bool) is 0 if False and 1 if true. So, int(header and remove_headers) will be 1 only if header and remove_headers are True. If they are True, the header is removed.

    return matrix


def write_csv(path: str, matrix, append=False):
    """
    Create a CSV file from a matrix

    :param path: path of the CSV file
    :type path: str
    :param matrix: matrix to write
    :type matrix: list of lists
    :return: the matrix containing data
    :param append: True if file must not be overwritten
    :type append: bool
    :rtype: None
    """

    with open(path, 'w' if not append else 'a') as file:
        writer(file, Dialect.delimiter).writerows([[str(col) for col in row] for row in matrix])


def export_running_time(label=None, time=None, unit="seconds", new_file=False):
    """
    Create a CSV file containing running times

    :param label: Label for the value
    :param time: Value
    :param unit: Unit for the value
    :param new_file: True if previous values must be overwritten
    """

    headers = ['label', 'running time', 'unit']
    file_path = join(path.get_csv_folder_path(), consts.RUNNING_TIMES)

    if new_file or not isfile(file_path):
        write_csv(file_path, [headers], append=False)
    if label is not None and time is not None:
        write_csv(file_path, [[str(label), str(time), str(unit)]], append=True)


def prediction_name_refactor(prediction_name):
    """
    Make prediction names more legible

    :param prediction_name: name of a prediction task
    :return: the refactored name
    """

    refactored_name = prediction_name
    if prediction_name == "['nb_solutions']":
        refactored_name = consts.OUTPUT_NB_SOLUTIONS
    elif prediction_name == "['single_solution']":
        refactored_name = consts.OUTPUT_IS_SINGLE_SOLUTION
    elif prediction_name == "['nb_solution_classes']":
        refactored_name = consts.OUTPUT_NB_SOLUTION_CLASSES
    elif prediction_name == "['single_solution_class']":
        refactored_name = consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES
    elif prediction_name == "['imbalance_count']":
        refactored_name = consts.OUTPUT_GRAPH_IMBALANCE_COUNT
    elif prediction_name == "['imbalance_percentage']":
        refactored_name = consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE
    return refactored_name


class ProgressBar:
    """
    This class displays and handles a CLI progress bar
    """

    def __init__(self, full_progress_number, bar_size=48, file=stdout):
        """
        Constructs a newly allocated ProgressBar instance

        :param full_progress_number: Value the counter has to reach to complete progress
        :param bar_size: Size of the progress bar
        :param file: Output file
        """

        self.full_progress_number = full_progress_number
        self.progress_counter = 0
        self.progress_percent = 0
        self.bar_size = bar_size
        self.file = file

    def initialize(self):
        """
        Display a 0% progress bar
        """

        self.progress_counter = 0
        self.progress_percent = 0
        print(" 0 %\t|", "".join(["-" for _ in range(self.bar_size)]), "|\r", sep="", end="", file=self.file)

    def update(self, counter_increment=1):
        """
        Update value of the progress bar

        Update value and print it if the bar or the printed percentage has grown

        :param counter_increment: value to add to the counter
        """
        self.progress_counter += counter_increment
        new_progress_percent = (self.progress_counter * 100) // self.full_progress_number
        if new_progress_percent > self.progress_percent:
            self.progress_percent = new_progress_percent
            progress_bar = (self.progress_percent * self.bar_size) // 100
            print(
                " {} %\t|{}{}|\r".format(
                    str(self.progress_percent),
                    "".join(["#" for _ in range(progress_bar)]),
                    "".join(["-" for _ in range(self.bar_size - progress_bar)])
                ), sep="", end="", file=self.file
            )

    def finalize(self):
        """
        Display a 100% progress bar
        """
        self.progress_counter = self.full_progress_number
        self.progress_percent = 100
        print(" 100 %\t|", "".join(["#" for _ in range(self.bar_size)]), "|", sep="", file=self.file)
