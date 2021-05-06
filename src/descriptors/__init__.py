#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains classes that compute a graph descriptors.

@author: Virgile Sucal
"""

import abc
from deprecated import deprecated


class GraphDescriptor(metaclass=abc.ABCMeta):
	"""
	Metaclass used as an interface for classes that contain methods computing graph descriptors

	There is an undirected() method to maintain compatibility with first version of CentralityMeasure.

	.. seealso: CentralityMeasure
	"""

	@classmethod
	def __subclasshook__(cls, subclass):
		return hasattr(subclass, 'undirected') and callable(subclass.undirected)

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
	def perform(graph, **kwargs):
		"""
		Compute the graph descriptor

		:param graph: the graph
		:type graph: igraph.Graph
		:param kwargs: hyper parameters
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError

	@staticmethod
	@abc.abstractmethod
	@deprecated("This function is deprecated, use 'perform()' instead")
	def undirected(graph, **kwargs):
		"""
		Compute the graph descriptor

		:param graph: the graph
		:type graph: igraph.Graph
		:param kwargs: hyper parameters
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError
