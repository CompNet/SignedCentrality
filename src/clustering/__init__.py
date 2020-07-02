#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package contains modules related to the clustering problem.

.. warning: In order to use this package, one has to install scikit-learn library.

.. seealso: scikit-learn
"""


class SVCKernel:
	"""
	Defines the names of the kernels that can be used in SVC.
	"""

	LINEAR = "linear"
	"""
	Linear kernel.
	"""

	POLY = "poly"
	"""
	Polynomial kernel.
	"""

	RBF = "rbf"
	"""
	Radial Basis Function (RBF) kernel
	"""

	SIGMOID = "sigmoid"
	"""
	Sigmoid kernel.
	"""

	PRECOMPUTED = "precomputed"
	"""
	Precomputed kernel.
	"""
