#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the module clustering.
"""

import unittest
from os.path import abspath, dirname

from signedcentrality import eigenvector_centrality, degree_centrality
from signedcentrality._utils.utils import *
from tests import load_data
from tests.clustering_test import Path
from subprocess import call
from glob import glob
from csv import reader, Sniffer
from igraph import Graph
from numpy import array


class ClusteringTest(unittest.TestCase):
	def __init__(self, method_name: str = ...) -> None:
		super().__init__(method_name)

		# self.data = load_data()
		load_data(Path.RES, Path.R_SCRIPT)

	def stub(self):
		self.assertEqual(True, False)


if __name__ == '__main__':
	unittest.main()
