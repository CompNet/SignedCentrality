#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains unit tests for the module classifier.

.. seealso: clustering
.. seealso: classifier
"""

from os import getcwd
import tests
import signedcentrality.clustering as clustering


class Path (clustering.Path):

	TESTS_RES_PATH = "../../res"
	"""
	Path to resource folder

	This path is defined in relation to the default path of unit tests working directory.
	It is used to change this working directory to res folder.
	"""

	FULL_DATASET_RES_PATH = "../../../res"
	"""
	Path to resource folder containing full dataset

	The dataset is out of the repository because it would be longer to complete for the IDE.
	"""

	DATASET_PATH = getcwd() + "/../../../res/clustering_dataset/inputs"
	"""
	Directory containing full dataset for clustering module

	The dataset is out of the repository because it would be longer to complete for the IDE.
	"""
