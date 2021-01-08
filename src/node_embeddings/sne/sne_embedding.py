#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the SNE computing.

The embedding is computed by following the method of S.Yuan, X. Wu and Y. Xiang.

.. note: S. Yuan, X. Wu and Y. Xiang. "SNE: signed network embedding". In: Pacific-Asia conference on knowledge discovery and data mining. 2017, p. 183-195. doi :10.1007/978-3-319-57529-2_15.
"""
from argparse import ArgumentParser
from logging import basicConfig, INFO
from os.path import dirname, abspath, exists
from pathlib import Path
from random import Random
import tensorflow.compat.v1 as tf
from os import system, makedirs
from centrality import CentralityMeasure
from node_embeddings.sne.sne.SNE import SNE, Options, FLAGS
from node_embeddings.sne.sne.walk import write_walks_to_disk, load_edgelist
from util import get_matrix


class SNEEmbedding(CentralityMeasure):
	"""
	This class is used to compute SNE

	Currently, this class extends CentralityMeasure class because it will be used in the same way as centrality measures.

	This processing is done in a class because it is used in the classifier.
	This classifier calls a method "undirected()" for all centrality computing classes which are in the package centrality.
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

	SNE_PATH = EMBEDDINGS_PATH + '/sne'
	"""
	Path to the sne directory
	"""

	MAIN_PATH = dirname(abspath(__file__)) + '/../../../out/embeddings/sne'
	"""
	Path to read and write files for SNE computing
	"""

	SAVE_PATH = MAIN_PATH + '/save_path'
	"""
	Path to save the model.
	"""

	DATA = MAIN_PATH + "/data"
	"""
	Path to write the files containing graph data to be read by SNE class.
	"""

	TRAIN_DATA = DATA + "/train_data.txt"
	"""
	Path to the training text file
	"""

	LABEL_DATA = DATA + "/label_data.txt"
	"""
	Path to the nodes labels text file
	"""

	WALKS_DATA = DATA + "/walks_data.txt"
	"""
	Path to the random walks on data
	"""

	# EMBEDDING_SIZE = 100
	EMBEDDING_SIZE = 10
	"""
	Embedding dimension size
	"""

	# SAMPLES_TO_TRAIN = 25
	SAMPLES_TO_TRAIN = 1
	"""
	Number of samples to train
	
	The unit is the million.
	"""

	LEARNING_RATE = 0.025
	"""
	Initial learning rate
	"""

	# NUM_SAMPLED = 512
	NUM_SAMPLED = 10
	"""
	Number of classes to randomly sample per batch
	"""

	CONTEXT_SIZE = 3
	"""
	Number of context nodes 
	"""

	BATCH_SIZE = 50
	"""
	Number of training examples processed per step
	"""

	IS_TRAIN = True
	"""
	True if the embedding need to be trained
	
	It have to always be True.
	"""

	NUMBER_WALKS = 20
	"""
	Number of random walks
	"""

	WALK_LENGTH = 40
	"""
	Length of a random walk
	"""

	@staticmethod
	def __initialize_directories():
		"""
		Create files and directories if they don't already exist
		"""
		if not exists(SNEEmbedding.TRAIN_DATA) or not exists(SNEEmbedding.LABEL_DATA):
			if not exists(SNEEmbedding.DATA):
				if not exists(SNEEmbedding.SNE_PATH):
					if not exists(SNEEmbedding.EMBEDDINGS_PATH):
						if not exists(SNEEmbedding.OUT_PATH):
							makedirs(SNEEmbedding.OUT_PATH)
						makedirs(SNEEmbedding.EMBEDDINGS_PATH)
					makedirs(SNEEmbedding.SNE_PATH)
				makedirs(SNEEmbedding.DATA)
			Path(SNEEmbedding.TRAIN_DATA).touch()
			Path(SNEEmbedding.LABEL_DATA).touch()

	@staticmethod
	def export_graph(graph):
		"""
		Export the graph in order to make it readable by SNE module

		:param graph: the graph
		:type graph: igraph.Graph
		"""

		# Create list:
		matrix = get_matrix(graph).toarray()
		length_range = range(len(matrix))
		adj_list = []
		for i in length_range:
			for j in length_range:
				if matrix[i][j] != 0:
					adj_list.append([i, j, int(matrix[i][j])])

		# Write files:
		SNEEmbedding.__initialize_directories()

		with open(SNEEmbedding.TRAIN_DATA, 'w+') as file:
			file.writelines([
				str("\t".join([
					str(col) for col in row
				]) + "\n") for row in adj_list
			])

		with open(SNEEmbedding.LABEL_DATA, 'w+') as file:
			file.writelines([
				str("\n".join([
					"\t".join([
						str(i), str(graph.vs[i]["id"])
					]) for i in range(graph.vcount())
				]))
			])

	@staticmethod
	def random_walk():
		"""
		Compute random walks

		The walks are computed for the graph which is given in static attribute TRAIN_DATA.
		"""

		write_walks_to_disk(
			load_edgelist(SNEEmbedding.TRAIN_DATA),
			f=SNEEmbedding.WALKS_DATA,
			num_paths=SNEEmbedding.NUMBER_WALKS,
			path_length=SNEEmbedding.WALK_LENGTH,
			alpha=0,
			rand=Random(1024)
		)

	@staticmethod
	def undirected(graph, scaled=False):
		"""
		Compute the SNE.

		This function reuse a part of SNE module code.

		:param graph: the graph
		:type graph: igraph.Graph
		:return: the embedding
		:param scaled: indicates if the embedding must be scaled
		:type scaled: bool
		:rtype: list
		"""

		# Flags Initialization:
		# tf.app.flags._global_parser = ArgumentParser()
		flags = tf.app.flags
		flags.DEFINE_string("save_path", SNEEmbedding.SAVE_PATH, "Directory to write the model and training sammaries.")
		flags.DEFINE_string("train_data", SNEEmbedding.TRAIN_DATA, "Training text file.")
		flags.DEFINE_string("label_data", SNEEmbedding.LABEL_DATA, "Nodes labels text file.")
		flags.DEFINE_string("walks_data", SNEEmbedding.WALKS_DATA, "Random walks on data")
		flags.DEFINE_integer("embedding_size", SNEEmbedding.EMBEDDING_SIZE, "The embedding dimension size.")
		flags.DEFINE_integer("samples_to_train", SNEEmbedding.SAMPLES_TO_TRAIN, "Number of samples to train(*Million).")
		flags.DEFINE_float("learning_rate", SNEEmbedding.LEARNING_RATE, "Initial learning rate.")
		flags.DEFINE_integer("num_sampled", SNEEmbedding.NUM_SAMPLED, "The number of classes to randomly sample per batch.")
		flags.DEFINE_integer("context_size", SNEEmbedding.CONTEXT_SIZE, "The number of context nodes .")
		flags.DEFINE_integer("batch_size", SNEEmbedding.BATCH_SIZE, "Number of training examples processed per step.")
		flags.DEFINE_boolean("is_train", SNEEmbedding.IS_TRAIN, "Train or restore")
		SNE.FLAGS = flags.FLAGS

		# Graph Export:
		SNEEmbedding.export_graph(graph)

		# Random Walk:
		SNEEmbedding.random_walk()

		# SNE TRAINING:
		emb_vertex = None
		basicConfig(level=INFO)
		with tf.Graph().as_default(), tf.Session() as session:
			model = SNE(Options(), session)
			model.train()
			emb_vertex = model.get_sne()

		return emb_vertex
