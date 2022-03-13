#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains modules related to node embeddings.

Each module of the package contains classes that compute a node embedding.

@author: Virgile Sucal

.. warning: In order to use this package, one has to install igraph and scipy libraries.
.. warning: The code has been tested with Python 3.8.7 (v3.8.7:6503f05dd5).

.. seealso: igraph
.. seealso: scipy
"""


import abc


class NodeEmbedding(metaclass=abc.ABCMeta):
	"""
	Metaclass used as an interface for classes that contain methods computing node embeddings

	There is an undirected() method to maintain compatibility with CentralityMeasure.

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
	def undirected(graph, **kwargs):
		"""
		Compute the node embedding

		:param graph: the graph
		:type graph: igraph.Graph
		:param kwargs: hyper parameters
		:return: the centrality
		:rtype: list
		"""

		raise NotImplementedError
