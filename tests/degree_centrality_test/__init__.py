#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains unit tests for the module degree_centrality.

.. seealso: centrality
.. seealso: degree_centrality
"""

import tests


class Path:
	R_RES = "./res/generated/R/*.csv"
	"""
	Directory containing test resources for clustering module
	"""

	R_SCRIPT = "degree_centrality_test.R"
	"""
	Script computing data to load in clustering_test module.
	"""
