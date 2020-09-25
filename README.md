# SignedCentrality

Centrality measures and prediction tasks for signed networks

* ?? Virgile Sucal ??
* Nejat Arinik [nejat.arinik@univ-avignon.fr](mailto:nejat.arinik@univ-avignon.fr)
* Vincent Labatut [vincent.labatut@univ-avignon.fr](mailto:vincent.labatut@univ-avignon.fr)
* Rosa Figueiredo [rosa.figueiredo@univ-avignon.fr](mailto:rosa.figueiredo@univ-avignon.fr)



## Description

This set of `Python` scripts/modules is designed for two main purposes:

* implement centrality measures for signed networks
* perform some prediction tasks, such as classification and regression, based on the features extracted from signed centrality measures or other graph-related measures/statistics. 



## Data

In order to compute centrality measures or to perform some prediction tasks, we include a sample dataset in the folder `in`. It is a part of the full training dataset used in our article: *Space of optimal solutions of the Correlation Clustering problem* (Arinik, Nejat; Labatut, Vincent, Figueiredo, Rosa (2020): Space of optimal solutions of the Correlation Clustering problem on Complete Signed Graphs. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v5).



## Organization

* Folder `in`: input signed networks.

* Folder `out`: contains the file produced by our scripts

* Folder `src`: 

  * Folder `centrality`: this folder contains signed centrality measures.
  * Folder `collect`: the scripts of this folder aims at collecting the statistics/measures which are already computed and stored in folders.
  * Folder `graph_embeddings`: this folder contains the methods calculating some features based on signed graph embeddings, i.e. at graph level.
  * Folder `node_embeddings`:  this folder contains the methods calculating some features based on signed node embeddings, i.e. at node level.
  * Folder `prediction`: this folder contains the scripts performing the prediction tasks, such as classification and regression.
  * Folder `stats`: this folder contains some graph-related statistics, e.g. spectral, structural, structural balance.

* Folder `tests`: 

  * Folder `centrality`:  All the unit and integration tests related to signed centrality measures.
  * Folder `stats`: All the unit and integration tests related to graph-related statistics.

  * Folder `graph_embeddings`: All the unit and integration tests related to signed graph embeddings.
  * Folder `node_embeddings`: All the unit and integration tests related to signed node embeddings.



## Installation

* Install Python (the tested version is Python 3.7.4)

* Install the following python modules
  * igraph 0.8.2
  * numpy 1.17.2
  * scipy 1.3.1
  * sklearn 0.21.3
  * pandas 0.25.1

* Download this project from GitHub

  You also need to retrieve the data from Figshare. Download and untar `Input Signed Networks.tar.gz` and `Evaluation Results.tar.gz`.  Place the contents of them into folders `in` and `output`, respectively. Finally, configure the input parameters in `src/main.py` and then run it inside the folder `src`.



## Dependencies

All the modules listed below are the python modules

* igraph 0.8.2
* numpy 1.17.2
* scipy 1.3.1
* sklearn 0.21.3
* pandas 0.25.1



## To-do list

* add more signed centrality measures
* add more graph-related statistics
* perform/analyze the classification task in depth
* perform/analyze the regression task in depth
* add some signed node embeddings
* add some signed graph embeddings
* add/update unit and integration tests