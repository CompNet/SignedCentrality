'''
Created on Sep 23, 2020

@author: nejat
'''

import itertools as iter
import numpy as np
import math

import consts

from csv import reader, Sniffer
from sys import float_info
from igraph import Graph


def format_4digits(x):
    """This method formats the given number (either integer or float)
        by forcing to have 4 digits.
       
    :param x: number
    :type x: int or float
    """
    return ('%.4f' % x)


def which(values):
    """This program finds the indexes of the elements that are True.
    Logically, the type of 'values' has to be boolean.
       
    values: boolean values
    """
    for value in values:
        if not isinstance(value,(bool,np.bool_)):
            raise Exception('which() function expects only boolean values')
    indxs = [indx for indx, bool_elt in enumerate(values) if bool_elt]
    return indxs
           
           
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
    nk = math.floor(n/l0)
    module_sizes = [nk] * l0
    nb_remaining = n - (nk * l0)
    for i in range(nb_remaining):
        module_sizes[i] += 1
    for i in range(l0):
        membership.extend([i+1] * module_sizes[i])
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
    module_sizes = [membership.count(indx) for indx in range(1,l0+1)]
    for m1, m2 in iter.combinations(range(1,l0+1), 2):
        n1 = module_sizes[m1-1]
        n2 = module_sizes[m2-1]
        prop_neg += (n1*n2)/((n*(n-1)/2))
    
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

