#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains unit tests for the module classifier.

.. seealso: clustering
.. seealso: classifier
"""

import tests
import clustering


class Path (clustering.Path):
	# # R_SCRIPT = "clustering_test.R"
	# R_SCRIPT = "../../src/clustering/graph_descriptors.R"
	# """
	# Script computing data to load in clustering_test module.
	# """

	DATASET_PATH = "../../../../clustering_dataset/"
	"""
	Directory containing full dataset for clustering module
	
	The dataset is out of the repository because it would be longer to complete for the IDE.
	"""

	TESTS_RES_PATH = "../../res"
	"""
	Path to resource folder
	
	This path is defined in relation to the default path of unit tests working directory.
	It is used to change this working directory to res folder.
	"""
