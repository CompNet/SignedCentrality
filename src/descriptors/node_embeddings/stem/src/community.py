import hdbscan
import models
import numpy as np
import sklearn.manifold as manifold
import matplotlib.pyplot as plt

def cluster_from_model(model: models.GraphEmbeddingModel, min_cluster_size=5, min_samples=None):
    vectors = model.get_all_weights()
    relevant_vectors = vectors[1:]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    if min_samples is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(relevant_vectors)
    return clusterer, cluster_labels


def plot_vectors(vectors, n_components=2):
    tsne = manifold.TSNE(n_components=n_components)
    projected_vectors = tsne.fit_transform(vectors)




def evaluate_correctness(X, y, model, kmeans):
    pass
