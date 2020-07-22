#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains modules related to the measures of centrality.

Each module of the package contains functions that compute a measure of centrality.

.. warning: In order to use this package, one has to install igraph and scipy libraries.

.. seealso: _utils
.. seealso: eigenvector_centrality
.. seealso: igraph
.. seealso: scipy
"""

import abc


class CentralityMeasure(metaclass=abc.ABCMeta):
	"""
	Metaclass used as an interface for classes that contain methods computing centrality measures

	The classifier only process undirected graphs.
	So, the only required method is undirected().
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
	def undirected(graph, scaled=False):
		"""
		Compute the centrality measure on an undirected graph

		:param graph: the graph
		:type graph: igraph.Graph
		:param scaled: indicates if the centrality must be scaled
		:type scaled: bool
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError
