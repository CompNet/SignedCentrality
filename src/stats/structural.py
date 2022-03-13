'''
Created on Sep 23, 2020

@author: nejat
'''

import numpy as np
import util


def retreive_pos_prop(g):
    """
    This method compute the ratio of positive weights to total weights.
       
    :param g: graph
    :type g: igraph.Graph
    :return: the v weight ratio
    :rtype: float
    """

    edge_weights = [e['weight'] for e in g.es]
    tot_sum = sum([abs(w) for w in edge_weights])
    pos_sum = sum([w for w in edge_weights if w>0])
    return pos_sum/tot_sum


def retreive_neg_prop(g):
    """
    This method compute the ratio of negative weights to total weights.
       
    :param g: graph
    :type g: igraph.Graph
    :return: the negative weight ratio
    :rtype: float
    """

    edge_weights = [e['weight'] for e in g.es]
    tot_sum = sum([abs(w) for w in edge_weights])
    neg_sum = abs(sum([w for w in edge_weights if w<0]))
    return neg_sum/tot_sum


def retreive_pos_neg_ratio(g):
    """
    This method computes the the positive/negative weight ratio
       
    :param g: graph
    :type g: igraph.Graph
    :return: the positive/negative ratio
    :rtype: float
    """

    ratio = retreive_pos_prop(g)/retreive_neg_prop(g)
    return ratio

