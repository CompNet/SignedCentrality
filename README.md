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

* There are 4 main files.

  * `main_feature_and_output_extraction.py`: It processes the features and the outputs.
  * `main_binary_classification.py`: It performs the binary classification task.
  * `main_ordinal_classification.py`: It performs the ordinal classification task.
  * `main_regression.py`: It performs the regression task.



## Installation

* Install Python (tested with Python 3.8.12)

* Install dependencies using the following command:

  ```
  pip install -r requirements.txt
  ```

* Download this project from GitHub

* You need to retrieve the data from [Figshare](https://doi.org/10.6084/m9.figshare.19350284). Download and untar `Input Signed Networks.tar.gz` and `Evaluation Results.tar.gz`.  Place them into the `in` and `output/evaluate-partitions` folders, respectively. Finally, we might need to configure the input parameters, i.e. the global variables such as `GRAPH_SIZES`, in the main files. You do not need to change these input parameters, if you work with the whole dataset. 
  


## How to run ?

### Use Case 1: The whole workflow -> 1) Extracting features and output variables, 2) Performing prediction tasks

* Go to the folder `src`.

* We run the file `main_feature_and_output_extraction.py` inside the folder `src`. Or, you can configure the `PYTHONPATH` variable, if you do not want to run it from the folder `src`. Note that it can take several hours.

* Apply the use case 2



### Use Case 2: Performing prediction tasks

* If you skip the use case 1, then you need to download and untar `csv.tar.gz` from [Figshare](https://doi.org/10.6084/m9.figshare.19350284). Place it into the `out/csv` folder.

* Go to the folder `src`.

* We run the files `main_binary_classification.py`, `main_ordinal_classification.py` and `main_regression.py` inside the folder `src`. Or, you can configure the `PYTHONPATH` variable, if you do not want to run it from the folder `src`.




## To-do list

* add more signed centrality measures
* add more graph-related statistics
* add some signed node embeddings
* add some signed graph embeddings
* add ordinal logistic regression from the package [statsmodels](https://www.statsmodels.org/devel/examples/notebooks/generated/ordinal_regression.html)
* add recursive feature selection method from the package [scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)
