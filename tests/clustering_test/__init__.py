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

	# R_SCRIPT = "clustering_test.R"
	R_SCRIPT = "../../src/clustering/graph_descriptors.R"
	"""
	Script computing data to load in clustering_test module.
	"""

	DATASET_SAMPLE_PATH = "../../res/clustering_dataset_sample/"
	"""
	Directory containing small dataset to test clustering module
	"""

	DATASET_PATH = "../../../../clustering_dataset/"
	"""
	Directory containing full dataset for clustering module
	
	The dataset is out of the repository because it would be longer to complete for the IDE.
	"""

	INPUTS_PATH = DATASET_SAMPLE_PATH + "inputs/"
	"""
	Directory containing full input dataset for clustering module
	"""

	RESULTS_PATH = DATASET_SAMPLE_PATH + "results/"
	"""
	Directory containing all results for INPUTS_PATH files
	"""

	CSV_EXT = ".csv"
	"""
	Pattern to use .csv files
	"""

	XML_EXT = ".xml"
	"""
	Pattern to use .csv files
	"""

	GRAPHML_EXT = ".graphml"
	"""
	Pattern to use .graphml files
	"""

	G_EXT = ".G"
	"""
	Pattern to use .G files
	"""

	NET_EXT = ".net"
	"""
	Pattern to use .net files
	"""

	DEFAULT_EXT = GRAPHML_EXT
	"""
	Default pattern
	"""

	GENERATED_CSV_PATHS_FILE = clustering.Path.GENERATED_RES_PATH + "/inputs.csv"
	"""
	Path of the CSV file containing the paths of the files which have to be used as input and results files in graph_descriptors.R.
	"""

	GENERATED_XML_PATHS_FILE = clustering.Path.GENERATED_RES_PATH + "/inputs.xml"
	"""
	Path of the XML file containing the paths of the files which have to be used as input and results files in graph_descriptors.R.
	"""
