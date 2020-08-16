# SignedCentrality

Centrality measures for signed networks.

## Run SignedCentrality

The only part of this program which is designed to be run is the `signedcentrality.clustering.classifier_comparison` module.
The other modules should be run only from unit tests.

### Run classifier comparison

To run `signedcentrality.clustering.classifier_comparison` module, one just have to run it from the directory which contains the module.

### Run tests

In Terminal, unit tests must be run from the directory containing the module where the tests are located.

In PyCharm, unit tests must be run from the Python class which extends `unittest.TestCase`.
The "main" code is only designed to run the test modules from Terminal.

## Use datasets to train classifiers

In order to train and test classifiers, there is a sample dataset. 
It is a part of the full training dataset _Space of optimal solutions of the Correlation Clustering problem_ (Arinik, Nejat; Labatut, Vincent (2019): _Space of optimal solutions of the Correlation Clustering problem_. figshare. Dataset. https://doi.org/10.6084/m9.figshare.8233340.v3).

To use full dataset, one should put it out of the repository, in a directory which the path should be `../res/clustering_dataset/inputs` from repository root. 
Then, `Path.load()` method must be set as `Path.load(DEFAULT_SAMPLE_INPUTS_PATH=Path.DATASET_PATH)` in `tests.clustering_test.clustering_test.ClusteringTest.__init__()` method or in `signedcentrality.clustering.classifier_comparison` "main" code.
To put full dataset in the repository might slow down the IDE.

If graph descriptors have been already computed, one should set `compute_descriptors` attribute to `False` in `ClassifierComparator` initialization.
