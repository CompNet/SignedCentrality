#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains utility modules that are used in the signedcentrality package.

These modules should not be used out of signedcentrality, except for tests.

.. note: The package name contains a leading underscore to prevent it from external use.
.. seealso: signedcentrality
"""


GRAPHML = "graphml"
"""
String defining GraphML format for the parameter "format" in method Graph.Read().
"""


WEIGHT = "weight"
"""
String defining GraphML id for the attribte that defines the weight of an edge.

This has to be given in the method graph.get_adjacency_sparse() which returns a weighted adjacency matrix for the given graph.
"""


ID = "id"
"""
String defining GraphML id for the attribte that defines the id of an edge.
"""
