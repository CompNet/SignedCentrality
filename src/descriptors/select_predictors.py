#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is used to select predictors to use in the program

@author: Virgile Sucal
"""

import importlib
import sys
from deprecated import deprecated
from descriptors import GraphDescriptor
from descriptors.centrality.degree_centrality import NegativeCentrality, PositiveCentrality, PNCentrality
from descriptors.centrality.eigenvector_centrality import EigenvectorCentrality
from descriptors.centrality.srwr_centrality import SRWRCentrality
from descriptors.centrality.trolltrust_centrality import TrollTrust
from descriptors.centrality.diversity_coef_centrality import diversity_coef_centrality
from descriptors.node_embeddings.sne.sne_embedding import SNEEmbedding
from descriptors.node_embeddings.sine.sine_embedding import SiNEEmbedding
from descriptors.node_embeddings.stem.stem_embedding import StEMEmbedding
from descriptors.centrality.node_effect_centrality import *


# Graph descriptors
__NOT_IN_GRAPH_DESCRIPTORS = [  # These classes aren't used as descriptors in models.
	PositiveCentrality,
	NegativeCentrality,
	# SRWRCentrality,
	TrollTrust,
	NodeEffect,
]


def __add_descriptor_classes(graph_descriptor_class):
	"""
	Select which descriptors will be computed and used

	Content of __NOT_IN_GRAPH_DESCRIPTORS won't be added.

	:param graph_descriptor_class: Superclass of all descriptors
	:return: dict containing descriptors
	"""

	graph_descriptors_dict = {}
	for subclass in graph_descriptor_class.__subclasses__():
		if subclass not in __NOT_IN_GRAPH_DESCRIPTORS:
			if len(subclass.__subclasses__()) > 0:
				graph_descriptors_dict = {**graph_descriptors_dict, **__add_descriptor_classes(subclass)}
			else:
				graph_descriptors_dict = {
					**graph_descriptors_dict,
					subclass.__name__: subclass.perform,
				}

	return graph_descriptors_dict


GRAPH_DESCRIPTORS = {
	**__add_descriptor_classes(GraphDescriptor)
}
