#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the StEM computing.

The embedding is computed by following the method of I. Rahaman and P. Hosein.

.. note: I. Rahaman and P. Hosein. "A method for learning representations of signed networks". In :14th International Workshop on Mining and Learning with Graphs. 2018, p. 31. doi :doi.org/10.475/123_4.
"""


from os.path import abspath, dirname
from descriptors import GraphDescriptor
from descriptors.node_embeddings.stem.stem.graph import *
from descriptors.node_embeddings.stem.stem.dataloaders import *
from descriptors.node_embeddings.stem.stem.models import PseudoKernelRankingModel, fit_ranking_model
from util import write_csv, get_matrix, get_adj_list
import descriptors.node_embeddings.stem.stem.dataloaders as dataloaders
import descriptors.node_embeddings.stem.stem.models as models
import descriptors.node_embeddings.stem.stem.classifiers as classifiers
import sklearn.linear_model as linear_model
import torch


class StEMEmbedding(GraphDescriptor):
	"""
	This class is used to compute StEM

	This processing is done in a class because it is used in the classifier.
	This classifier calls a method "undirected()" for all centrality computing classes which are in the package centrality.
	"""

	ROOT_PATH = dirname(abspath(__file__)) + '/../../../..'
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

	STEM_PATH = EMBEDDINGS_PATH + '/stem'
	"""
	Path to the StEM directory
	"""

	SAVE_PATH = STEM_PATH + '/save_path'
	"""
	Path to save the model.
	"""

	DATA = STEM_PATH + "/data"
	"""
	Path to write the files containing graph data to be read by StEM classes.
	"""

	TRAIN_DATA = DATA + "/soc-sign-Slashdot090221.txt"
	"""
	Path to the training file.
	"""

	GENERATED_INPUT_DATA = DATA + "/train_data.csv"
	"""
	Path to the generated CSV file.
	"""

	# EMBEDDING_SIZE = 32  # Default value in paper
	EMBEDDING_SIZE = 10  # To have the same size in all embeddings
	"""
	Embedding dimension size.
	"""

	@staticmethod
	def perform_csv(file=GENERATED_INPUT_DATA, **kwargs):
		"""
		Train StEM embedding

		This function reuse the code of file "KernelRanking Test.ipynb" in StEM code.

		:param file: path for the graph
		:type file: str
		:param kwargs: hyper parameters
		:return: the embedding
		:rtype: list
		"""

		# Create graph from CSV and extract datasets
		delimiter = ','
		ratio = .8
		if 'delimiter' in kwargs:
			delimiter = kwargs['delimiter']
		if 'ratio' in kwargs:
			ratio = kwargs['ratio']
		data = UnsplitDataset(file, delimiter=delimiter, ratio=ratio)  # ratio=1 means "no train set"
		triples, triples0 = data.get_training_triples(True)
		# print("triples =", triples)
		# print("triples0 =", triples0)
		batch_size = int(0.4 * len(triples))
		batch_size0 = int(0.4 * len(triples0))

		# Set hyper-parameters to values which have been selected by authors
		default_kwargs = {
			"num_nodes": data.get_num_nodes(),
			"dims": StEMEmbedding.EMBEDDING_SIZE,
			"triples": triples,
			"triples0": triples0,
			"batch_size": batch_size,
			"batch_size0": batch_size0,
			"delta": 1,
			"delta0": 0.5,
			"epochs": 150,
			"lr": 0.5,
			"lam": 0.00055,
			"lr_decay": 0.0,
			"p": 2,
			"print_loss": False,
			"p0": True,
		}

		train_kwargs = {}
		for key in kwargs.keys():
			if key in default_kwargs.keys():
				train_kwargs[key] = kwargs[key]

		for key in default_kwargs.keys():
			if not key in train_kwargs.keys():
				train_kwargs[key] = default_kwargs[key]

		train_kwargs['p0'] = len(triples0) > 0

		# num_nodes = data.get_num_nodes()
		# print("num_nodes =", num_nodes)
		# print_loss = False
		# epochs = 150
		# dims = StEMEmbedding.EMBEDDING_SIZE
		# lr = 0.5
		# lr_decay = 0.0
		# weight_decay = 0.0
		# lam = 0.00055
		# p = 2
		# delta = 1
		# delta0 = 0.5
		# if 'print_loss' in kwargs:
		# 	print_loss = kwargs['print_loss']
		# if 'epochs' in kwargs:
		# 	epochs = kwargs['epochs']
		# if 'dims' in kwargs:
		# 	dims = kwargs['dims']
		# if 'lr' in kwargs:
		# 	lr = kwargs['lr']
		# if 'lr_decay' in kwargs:
		# 	lr_decay = kwargs['lr_decay']
		# if 'weight_decay' in kwargs:
		# 	weight_decay = kwargs['weight_decay']
		# if 'lam' in kwargs:
		# 	lam = kwargs['lam']
		# if 'p' in kwargs:
		# 	p = kwargs['p']
		# if 'delta' in kwargs:
		# 	delta = kwargs['delta']
		# if 'delta0' in kwargs:
		# 	delta0 = kwargs['delta0']
		# dims_array = [dims, 20, 20]

		# Train model
		ranking_model = models.fit_ranking_model(
			**train_kwargs
			# num_nodes=num_nodes,
			# dims=dims_array[0],
			# triples=triples,
			# triples0=triples0,
			# delta=delta,
			# delta0=delta0,
			# batch_size=batch_size,
			# batch_size0=batch_size0,
			# epochs=epochs,
			# lr=lr,
			# print_loss=print_loss
		)

		# X, y = data.get_testing_set()
		#
		# # Train model
		# ranking_model = models.fit_pseudo_kernel_model(
		# 	num_nodes,
		# 	dims_array[0],
		# 	X=X,
		# 	y=y
		# )

		# Get embedding
		embedding = ranking_model.get_embeddings()
		embedding = torch.mean(embedding, dim=0)
		embedding = embedding.detach().numpy()

		return embedding, ranking_model

	@staticmethod
	def perform(graph, **kwargs):
		"""
		Train StEM embedding

		:param graph: the graph
		:type graph: igraph.Graph
		:param kwargs: hyper parameters
		:return: the embedding
		:rtype: list
		"""

		# Write CSV graph in GENERATED_INPUT_DATA as an edgelist.
		# Each line is "<source>,<target>,<sign>".
		write_csv(StEMEmbedding.GENERATED_INPUT_DATA, get_adj_list(graph))

		return StEMEmbedding.perform_csv(**kwargs)[0]


if __name__ == '__main__':
	# Test if StEM code is well integrated in project code:
	file = StEMEmbedding.TRAIN_DATA
	embedding, ranking_model = StEMEmbedding.perform_csv(file, delimiter='\t', print_loss=True)
	print("Embedding =", embedding)
	# print(len(embedding))


