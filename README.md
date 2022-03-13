# SignedCentrality

Centrality measures and prediction tasks for signed networks

* Nejat Arinik [nejat.arinik@inrae.fr](mailto:nejat.arinik@inrae.fr)
* Vincent Labatut [vincent.labatut@univ-avignon.fr](mailto:vincent.labatut@univ-avignon.fr)
* Rosa Figueiredo [rosa.figueiredo@univ-avignon.fr](mailto:rosa.figueiredo@univ-avignon.fr)



## Description

This set of `Python` scripts/modules is designed for two main purposes:

* implementing centrality measures for signed networks
* performing some prediction tasks, such as classification and regression, based on the features extracted from signed centrality measures or other graph-related measures/statistics. 



## Data

In order to compute centrality measures or to perform some prediction tasks, we include a sample dataset in the folder `in`. It is a part of the full training dataset used in our article: *Space of optimal solutions of the Correlation Clustering problem* (Arinik, Nejat; Labatut, Vincent, Figueiredo, Rosa (2020): Space of optimal solutions of the Correlation Clustering problem on Complete Signed Graphs. figshare. We also extended this dataset with incomplete signed networks. The final dataset can be found on [FigShare](https://doi.org/10.6084/m9.figshare.19350284).



## Organization

* Folder `in`: input signed networks.

* Folder `out`: contains the files produced by our scripts

* Folder `src`: 

  * Folder `descriptors`: this folder contains three groups of measures: 1) centrality, 2) node embeddings, and 3) graph embeddings.
  * Folder `collect`: the scripts of this folder aim at collecting the statistics/measures which are already computed and stored in folders.
  * Folder `prediction`: this folder contains the scripts performing the prediction tasks, such as classification and regression.
  * Folder `stats`: this folder contains some graph-related statistics, e.g. spectral, structural, structural balance.

* Folder `tests`: 

  * Folder `centrality`:  All the unit and integration tests related to signed centrality measures.
  * Folder `stats`: All the unit and integration tests related to graph-related statistics.
  * Folder `graph_embeddings`: All the unit and integration tests related to signed graph embeddings.
  * Folder `node_embeddings`: All the unit and integration tests related to signed node embeddings.



## Installation

* Install Python (tested with Python 3.8.12)

* Install dependencies using following command:

  ```
  pip install -r requirements.txt
  ```
  
* Download this project from GitHub

  You also need to retrieve the data from [Figshare](https://doi.org/10.6084/m9.figshare.19350284). Download and untar `Input Signed Networks.tar.gz` and `Evaluation Results.tar.gz`.  Place the contents of them into folders `in` and `output/evaluate-partitions`, respectively. Finally, configure the input parameters in one of the main files and then run it inside the folder `src`.

* The project must be run from `src/` directory.


* There are 4 main files.

  * `main_feature_and_output_extraction.py`: The main file, in which we process the features and the outputs.
  * `main_binary_classification.py`: The main file, in which we perform the binary classification task.
  * `main_ordinal_classification.py`: The main file, in which we perform the ordinal classification task.
  * `main_regression.py`: The main file, in which we perform the regression task.


## Dependencies

All the modules listed below are the python modules

* python-igraph 0.8.2
* numpy 1.19.2
* scipy 1.3.1
* scikit-learn 0.21.3
* pandas 0.25.1
* tensorflow 2.4.0
* torch 1.7.1
* deprecated 1.2.11
* hdbscan 0.8.24
* pony 0.7.14
* gem 1.0.1 from https://github.com/palash1992/GEM.git
* keras 2.0.2
* seaborn 0.11.1
* imbalanced-learn 0.9.0.dev0 from https://github.com/scikit-learn-contrib/imbalanced-learn.git
* rpy2 3.4.4
* tqdm 4.60.0
* bctpy 0.5.2


## To-do list

* add more signed centrality measures
* add more graph-related statistics
* add some signed node embeddings
* add some signed graph embeddings
* add ordinal logistic regression from the package [statsmodels](https://www.statsmodels.org/devel/examples/notebooks/generated/ordinal_regression.html)
* add recursive feature selection method from the package [scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)
