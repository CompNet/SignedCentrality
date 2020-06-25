#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the measures of degree centrality for signed graphs.

The measure is computed by following the method of Martin Everett and Stephen Borgatti.

.. note: M. Everett & S. Borgatti. (2014). Networks containing negative ties. SocialNetworks, 38, 111-120. 10.1016/j.socnet.2014.03.005
"""

import abc
from sys import float_info

from numpy import array, identity, dot, transpose
from numpy.linalg import inv
from signedcentrality._utils.utils import *


class DegreeCentrality(metaclass = abc.ABCMeta):
	"""
	Metaclass used as an interface for classes that contain methods computing measures of degree centrality
	"""

	@classmethod
	def __subclasshook__(cls, subclass):
		return hasattr(subclass, 'incoming') and callable(subclass.incoming) and hasattr(subclass, 'outgoing') and callable(subclass.outgoing) and hasattr(subclass, 'undirected') and callable(subclass.undirected)

	def __new__(cls, *args, **kwargs):
		"""
		Constructor

		This class, and its subclasses can't be instantiated.

		:param args: unused
		:param kwargs: unused
		"""

		raise TypeError("This class can't be instantiated.")

	@staticmethod
	@abc.abstractmethod
	def incoming(graph, scaled = False):
		"""
		Compute degree centrality on incoming edges

		:param graph: the graph
		:type graph: igraph.Graph
		:param scaled: indicates if the centrality must be scaled
		:type scaled: bool
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError

	@staticmethod
	@abc.abstractmethod
	def outgoing(graph, scaled = False):
		"""
		Compute degree centrality on outgoing edges

		:param graph: the graph
		:type graph: igraph.Graph
		:param scaled: indicates if the centrality must be scaled
		:type scaled: bool
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError

	@staticmethod
	@abc.abstractmethod
	def undirected(graph, scaled = False):
		"""
		Compute degree centrality on both incoming and outgoing edges

		:param graph: the graph
		:type graph: igraph.Graph
		:param scaled: indicates if the centrality must be scaled
		:type scaled: bool
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError


class PositiveCentrality(DegreeCentrality):
	"""
	Contain methods computing measures of degree centrality on graphs that contains only positive edges.

	This graph may be a signed graph without negative edges or an unsigned graph.
	"""

	@staticmethod
	def incoming(graph, scaled = False):
		return NegativeCentrality.incoming(graph, scaled)

	@staticmethod
	def outgoing(graph, scaled = False):
		return NegativeCentrality.outgoing(graph, scaled)

	@staticmethod
	def undirected(graph, scaled = False):
		return NegativeCentrality.undirected(graph, scaled)

class NegativeCentrality(DegreeCentrality):
	"""
	Contain methods computing measures of degree centrality on graphs that contains only negative edges.

	This graph may be a signed graph without positive edges or an unsigned graph which will be processed in the same way as the negative graph.
	Actually, both of these graphs will be treated as unsigned graphs representing a negative graph.
	If the graph is, in fact, a negative signed graph, it will be converted in positive graph.
	"""

	@staticmethod
	def incoming(graph, scaled = False):
		A = get_matrix(graph).toarray()
		n = len(A)
		I = identity(n)
		ones = array([1 for _ in range(n)])
		beta1 = 1. / (n - 1.)
		beta2 = 1. / ((n - 1.) ** 2)

		h_star = dot(dot(inv(I - beta2 * dot(transpose(A), A)), (I - beta1 * transpose(A))), ones)

		if not scaled:
			return h_star

		return scale_centrality(h_star)

	@staticmethod
	def outgoing(graph, scaled = False):
		A = get_matrix(graph).toarray()
		n = len(A)
		I = identity(n)
		ones = array([1 for _ in range(n)])
		beta1 = 1. / (n - 1.)
		beta2 = 1. / ((n - 1.) ** 2)

		h_star = dot(dot(inv(I - beta2 * dot(A, transpose(A))), (I - beta1 * A)), ones)

		if not scaled:
			return h_star

		return scale_centrality(h_star)

	@staticmethod
	def undirected(graph, scaled = False):
		A = get_matrix(graph).toarray()
		n = len(A)
		I = identity(n)
		ones = array([1 for _ in range(n)])
		beta1 = 1. / (n - 1.)

		h_star = dot(inv(I + beta1 * A), ones)

		if not scaled:
			return h_star

		return scale_centrality(h_star)



class PNCentrality(DegreeCentrality):
	"""
	Contain methods computing measures of degree centrality on graphs that contains both positive and negative edges.
	"""

	@staticmethod
	def incoming(graph, scaled = False):
		matrix = get_matrix(graph).toarray()
		n = len(matrix)
		I = identity(n)

		P = array([[max(col, 0) for col in row] for row in matrix])
		N = array([[min(col, 0) for col in row] for row in matrix])
		# A = P - 2. * N  # All weights
		A = P + 2. * N  # In the article, the formula is 'A = P - 2. * N'. The operator '+' is used here because N is already negative.

		ones = array([1 for _ in range(n)])
		beta1 = 1. / (4 * ((n - 1.) ** 2))
		beta2 = 1. / (2 * (n - 1.))

		PN = dot(dot(inv(I - beta1 * dot(transpose(A), A)), (I + beta2 * transpose(A))), ones)

		if not scaled:
			return PN

		return scale_centrality(PN)

	@staticmethod
	def outgoing(graph, scaled = False):
		matrix = get_matrix(graph).toarray()
		n = len(matrix)
		I = identity(n)

		P = array([[max(col, 0.) for col in row] for row in matrix])  # Positive weights
		print(P, end="\n\n")
		N = array([[min(col, 0.) for col in row] for row in matrix])  # Negative weights
		print(N, end="\n\n")
		# A = P - 2. * N  # All weights
		A = P + 2. * N  # In the article, the formula is 'A = P - 2. * N'. The operator '+' is used here because N is already negative.
		print(A, end="\n\n")

		ones = array([1 for _ in range(n)])
		beta1 = 1. / (4 * ((n - 1.) ** 2))
		beta2 = 1. / (2 * (n - 1.))

		PN = dot(dot(inv(I - beta1 * dot(A, transpose(A))), (I + beta2 * A)), ones)

		if not scaled:
			return PN

		return scale_centrality(PN)

	@staticmethod
	def undirected(graph, scaled = False):
		matrix = get_matrix(graph).toarray()
		n = len(matrix)
		I = identity(n)

		P = array([[max(col, 0.) for col in row] for row in matrix])  # Positive weights
		# print(P, end="\n\n")
		N = array([[min(col, 0.) for col in row] for row in matrix])  # Negative weights
		# print(N, end="\n\n")
		# A = P - 2. * N  # All weights
		A = P + 2. * N  # In the article, the formula is 'A = P - 2. * N'. The operator '+' is used here because N is already negative.
		# print(A, end="\n\n")

		ones = array([1 for _ in range(n)])
		beta1 = 1. / (2. * n - 2.)

		PN = dot(inv(I - beta1 * A), ones)

		if not scaled:
			return PN

		return scale_centrality(PN)
