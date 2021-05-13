# README #

Code for the paper "SNE: Signed Network Embedding"

### Set up ###

* Run `python walk.py` to generate random walks
* Run `python SNE.py` to train the network embeddings
* `test.py` includes the testing code for node classification and link prediction

### Data ###
*wiki_edit.txt* is the edge list file of signed graph "WikiEditor" described in paper

*wiki_usr_labels.txt* contains the labels of "WikiEditor" nodes

### Requirements ###

* Python 3.6
* Tensorflow 1.2.1