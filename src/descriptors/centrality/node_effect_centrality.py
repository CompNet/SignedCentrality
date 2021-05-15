#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the measure of A simple approach for quantifying node centrality
in signed and directed social networks centrality.

The measure is computed by following the method of Wei-Chung Liu, Liang-Cheng Huang, Chester Wai-Jen Liu & Ferenc Jordán.

.. note: WC. Liu, LC. Huang, C. WJ. Liu et J. Ferenc. «A simple approach for quantifying node centrality
in signed and directed social networks». In :Applied Network Science5.1 (août 2020), p. 46. issn : 2364-8228.
doi :10.1007/s41109-020-00288-w
"""

from descriptors import GraphDescriptor


class NodeEffect(GraphDescriptor):
    """
    This class is used to compute node effects centralities
    """

    @staticmethod
    def perform_all(graph, **kwargs):
        """
        Compute this centrality.
        """
        # todo ajouter code centralité


class NodeEffectTotalIndex(NodeEffect):
    """
    This class will return the value of the total effect of a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return total effect value
        """

class NodeEffectNetIndex(NodeEffect):
    """
    This class will return the value of the net effect exerted by a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return net effect exerted value
        """

class NodeEffectNetIndex1(NodeEffect):
    """
    This class will return the value of the net effect received by a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return net effet received value
        """
