#!/usr/bin/env python
# -*- coding: utf-8 -*-

from igraph import *
from scipy import *
import networkx
from collections import OrderedDict
from numpy import *
from scipy.sparse import lil_matrix
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.sparse import linalg

from signedcentrality._utils.utils import *

"""
This module contains functions related to the measure of eigenvector centrality.

The measure is computed by following the method of Phillip Bonacich and Paulette Lloyd.

.. note: P. Bonacich & P. Lloyd. (2004). Calculating status with negative relations. SocialNetworks, 26, 331-338. 10.1016/j.socnet.2004.08.007

"""


def diagonal(n1, n2):
	"""
	Create a diagonal squared matrix.

	Values are set as explained in the article written by P. Bonacich and P. Lloyd.
	Using scipy.sparse.lil_matrix is more efficient to contruct the matrix.
	Since it is to slow to compute arithmetic operations, it is converted in csr_matrix which is more efficient to such operations.

	:param n1: number of individuals in the first set
	:type n1: int
	:param n2: number of individuals in the second set
	:type n2: int
	:return: the matrix
	:rtype: scipy.sparse.csr_matrix
	"""

	diag = lil_matrix(numpy.array([numpy.array([0. for _1 in range(n1 + n2)]) for _0 in range(n1 + n2)]))  # Create a squared matrix of floats which is initialized with the float value -1.
	diag.setdiag(-1.)
	diag.setdiag([1. for _ in range(n1)])
	return diag.tocsr()


def compute_eigenvector_centrality(graph, D=None, scaled=False):
	"""
	Compute the eigenvector centrality.

	It is computed using the NetworkX Library.
	Indeed, the method computing the eigenvecor centrality in NetworkX is able to compute it correctly for signed graphs.

	:param graph: the graph
	:type graph: igraph.Graph
	:return: the eigenvector centrality
	:rtype: list
	"""

	scale = 1

	nx_graph = networkx.from_scipy_sparse_matrix(get_matrix(graph), False, None, WEIGHT)

	centrality = networkx.eigenvector_centrality_numpy(nx_graph, WEIGHT)  # The result isn't scaled.
	# centrality = networkx.eigenvector_centrality(nx_graph, 1000, 1.e-10, weight=WEIGHT)

	if scaled:
		max_value = sys.float_info.min  # Minimal value of a float
		for _, value in centrality.items():
			if value > max_value:
				max_value = value
		scale = 1 / max_value

	scaled_centrality = [value * scale for _, value in OrderedDict(sorted(centrality.items())).items()]

	return scaled_centrality


