#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the SiNE computing.

The embedding is computed by following the method of S. Wang, J. Tang, C. Aggarwal, Y. Chang and H. Liu. .

.. note: S. Wang, J. Tang, C. Aggarwal, Y. Chang and H. Liu. "Signed Network Embedding in SocialMedia". In :Proceedings of the 2017 SIAM International Conference on Data Mining(SDM), p. 327-335. doi :10.1137/1.9781611974973.37.
"""

from os.path import abspath, dirname
from descriptors import GraphDescriptor
from descriptors.node_embeddings.sine.sine.graph import *
from descriptors.node_embeddings.sine.sine.model import SiNE, fit_model
from util import write_csv, get_matrix


class SiNEEmbedding(GraphDescriptor):
	"""
	This class is used to compute SiNE

	This processing is done in a class because it is used in the classifier.
	This classifier calls a method "undirected()" for all centrality or embedding computing classes.
	"""

	ROOT_PATH = dirname(abspath(__file__)) + '/../../..'
	"""
	Path to the root directory
	"""

	OUT_PATH = ROOT_PATH + '/out'
	"""
	Path to the out directory
	"""

	EMBEDDINGS_PATH = OUT_PATH + '/embeddings'
	"""
	Path to the embeddings directory
	"""

	SNE_PATH = EMBEDDINGS_PATH + '/sine'
	"""
	Path to the SiNE directory
	"""

	SAVE_PATH = SNE_PATH + '/save_path'
	"""
	Path to save the model.
	"""

	DATA = SNE_PATH + "/data"
	"""
	Path to write the files containing graph data to be read by SNE class.
	"""

	TRAIN_DATA = DATA + "/soc-Epinions1.txt"
	"""
	Path to the training file.
	"""

	GENERATED_INPUT_DATA = DATA + "/train_data.csv"
	"""
	Path to the generated CSV file.
	"""

	EMBEDDING_SIZE = 10
	"""
	Embedding dimension size.
	"""

	D = 20
	"""
	Parameter d in the article.
	"""

	N = 3
	"""
	Parameter N in the article.
	"""

	ALPHA = 0.0001
	"""
	Parameter 𝛂 in the article.
	"""

	DELTA = 1.0
	"""
	Parameter 𝛅 in the article.
	"""

	DELTA_0 = 0.5
	"""
	Parameter 𝛅0 in the article.
	"""

	MINI_BATCH_SIZE = 300
	"""
	Size of the mini-batch.
	"""

	LAYER_INPUT_DIM = EMBEDDING_SIZE
	"""
	Number of dimensions for the input of hidden layers.
	"""

	LAYER_OUTPUT_DIM = EMBEDDING_SIZE
	"""
	Number of dimensions of hidden layers.
	"""

	EPOCHS = 30
	"""
	Number of epochs to train the model.
	"""

	@staticmethod
	def undirected_csv(file=GENERATED_INPUT_DATA, **kwargs):
		"""
		Train SiNE embedding

		:param file: path for the graph
		:type file: str
		:param kwargs: hyper parameters
		:return: the embedding
		:rtype: list
		"""

		# Create graph from CSV
		graph = Graph.read_from_file(file, True)

		# Create model
		model = SiNE(len(graph), SiNEEmbedding.LAYER_INPUT_DIM, SiNEEmbedding.LAYER_OUTPUT_DIM)

		# Train model
		fit_model(model, graph.get_triplets(), SiNEEmbedding.DELTA, SiNEEmbedding.MINI_BATCH_SIZE, SiNEEmbedding.EPOCHS, SiNEEmbedding.ALPHA)

		# Get embedding
		embedding = model.get_x()

		return embedding, model

	@staticmethod
	def undirected(graph, **kwargs):
		"""
		Train SiNE embedding

		:param graph: the graph
		:type graph: igraph.Graph
		:param kwargs: hyper parameters
		:return: the embedding
		:rtype: list
		"""

		# Write CSV graph in GENERATED_INPUT_DATA.
		write_csv(SiNEEmbedding.GENERATED_INPUT_DATA, get_matrix(graph).toarray())

		return SiNEEmbedding.undirected_csv(**kwargs)[0]


