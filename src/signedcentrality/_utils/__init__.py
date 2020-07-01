#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains utility modules that are used in the signedcentrality package.

These modules should not be used out of signedcentrality, except for tests.

.. note: The package name contains a leading underscore to prevent it from external use.
.. seealso: signedcentrality
"""


class Format:

	GRAPHML = "graphml"
	"""
	String defining GraphML format for the parameter "format" in method Graph.Read().
	"""

	CSV = "csv"
	"""
	String defining CSV format for the parameter "format" in method read_graph().
	"""


class FileIds:
	WEIGHT = "weight"
	"""
	String defining GraphML id for the attribute that defines the weight of an edge.

	This has to be given in the method graph.get_adjacency_sparse() which returns a weighted adjacency matrix for the given graph.
	"""

	SIGN = "sign"
	"""
	String defining GraphML id for the attribute that defines the weight of an edge.

	The weight is called "sign" in degree_centrality module.
	"""

	ID = "id"
	"""
	String defining GraphML id for the attribute that defines the id of an edge.
	"""
