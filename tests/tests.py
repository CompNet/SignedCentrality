#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
# import igraph
from igraph import *
from signedcentrality.eigenvector_cenrality import *
from signedcentrality._utils.utils import *

"""
This module contains unit tests for the modules of the package signedcentrality.
"""

class SignedCentralityTest(unittest.TestCase):
	# def test_success(self):
	# 	self.assertEqual(True, True)  # Stub
	#
	# def test_failure(self):
	# 	self.assertEqual(True, False)  # Stub

	def test_readGraph(self):
		self.assertIsInstance(readGraph("sampson.graphml"), Graph([(0, 1)]))


if __name__ == '__main__':
	unittest.main()
