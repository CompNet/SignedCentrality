#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from igraph import *
from signedcentrality._utils.utils import *

"""
This module contains functions related to the measure of eigenvector centrality.

The measure is computed by following the method of P. Bonacich and P. Lloyd.

.. note: P. Bonacich & P. Lloyd. (2004). Calculating status with negative relations. SocialNetworks, 26, 331-338. 10.1016/j.socnet.2004.08.007

"""


def diagonal(n1, n2):
	"""
	Create a diagonal squared matrix.

	Values are set as explained in the article written by P. Bonacich and P. Lloyd.

	:param n1: number of individuals in the first set
	:param n2: number of individuals in the second set
	:return: the matrix
	:rtype: scipy.sparse.csr_matrix
	"""

	diag = csr_matrix(numpy.array([numpy.array([0. for _1 in range(n1 + n2)]) for _0 in range(n1 + n2)]))  # Create a squared matrix of floats which is initialized with the float value -1.
	diag.setdiag(-1.)
	diag.setdiag([1. for _ in range(n1)])
	return diag


