#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the SNE tests.
"""

from consts import FILE_FORMAT_GRAPHML
from node_embeddings.sne.sne_embedding import *
from util import read_graph

if __name__ == '__main__':
	network_path = dirname(abspath(__file__)) + "/../../../in/n=20_l0=3_dens=1.0000/propMispl=0.2000/propNeg=0.7000/network=1/signed-unweighted.graphml"
	graph = read_graph(network_path, FILE_FORMAT_GRAPHML)
	emb_vertex = SNEEmbedding.undirected(graph)
	print(emb_vertex)
