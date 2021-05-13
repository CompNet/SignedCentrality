"""
Used to manage access to data sources
"""


import csv
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import  hamming
import pickle
import util
import graph
import re
import networkx as nx





class IdAssigner(object):
    def __init__(self):
        self._id2object = defaultdict(lambda: None)
        self._object2id = defaultdict(lambda: None)
        self.count = 0

    def insert(self, obj):
        """
        Inserts an object into the IdAssigner's records
        Parameters
        ----------
        obj : object
            The object to be inserted


        Returns
        -------
        int
            The id assigned to the object inserted

        """
        if obj not in self._object2id:
            self.count += 1
            self._object2id[obj] = self.count
            self._id2object[self.count] = obj
        return self._object2id[obj]

    def id2object(self, id):
        """
        Gets the object with the queried ID
        Parameters
        ----------
        id : int
            The ID being queried

        Returns
        -------
        object
            The object indexed by the queried ID if it exists, else None

        """
        return self._id2object[id]

    def object2id(self, obj):
        """
        Queries the ID for a particular object
        Parameters
        ----------
        obj : object
            The object whose ID is being requested


        Returns
        -------
        int
            The ID of the requested object

        """
        return self._object2id[obj]


def load_graph_data_as_numpy(filepath, delimiter=','):
    """
    Reads an edge list where each line is an edge pair together with a real number between
    -1 and 1 that indicates whether said edge negative or postive
    Parameters
    ----------
    filepath : str
        The path to the edge list
    delimiter : str (optional)
        The delimiter used to separate values in the file (default is ',')

    Returns
    -------
    array
        The edge pairs with each node replaced with its automatically assigned ID
    array
        The labels (0 is negative, 1 is positive)
    IdAssigner
        IdAssigner that can map IDs back to their original node names

    """
    id_assigner = IdAssigner()
    file = open(filepath, 'r')
    reader = csv.reader(file, delimiter=delimiter)
    X = [] # stores the edges as pairs
    y = [] # stores a sign label for edges (positive, i.e. 1, or negative, i.e. 0)
    for line in reader:
        u, v, score = line
        
        u_id = id_assigner.insert(u)
        v_id = id_assigner.insert(v)
        x = [u_id, v_id]
        
        #print(f'{line} --> {x}')
        score = int(float(score) * 10)
        if score > 0:
            y.append(1)
            X.append(x)
        elif score < 0:
            y.append(0)
            X.append(x)
        #print(f'{u},{v}')

    X = np.array(X)
    y = np.array(y)
    return X, y, id_assigner


class Dataset(object):
    def __init__(self):
        pass

    def get_training_set(self):
        raise NotImplementedError

    def get_testing_set(self):
        raise NotImplementedError

    def persist(self, filepath):
        """
        Pickles that dataset to facilitates reuse
        Parameters
        ----------
        filepath : str
            The path to which this Dataset is persisted

        Returns
        -------
        None

        """
        with open(filepath, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class UnsplitDataset(Dataset):
    def __init__(self, filepath, delimiter=',', ratio=0.8):
        """
        Creates an UnsplitDataset. Here we assume that we have the edge list for the
        entire graph and we are randomly dividing the edges into a training set and testing set
        Parameters
        ----------
        filepath : str
            The filepath to the edge list for the graph
        delimiter : str (optional)
            The delimiter used to separate values in the file (default is ',')
        ratio : float (optional)
            The training-testing split ratio. Specifies what percentage of rows to use
            in the training set (default is 0.8)
        """
        super(Dataset, self).__init__()
        self.filepath = filepath
        with open(filepath, 'r') as fp:
            self.X, self.y, self.id_assigner = load_graph_data_as_numpy(filepath, delimiter)
            self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y,
                                                                                  train_size=ratio,
                                                                                    test_size=1-ratio,
                                                                                    shuffle=True)
        self.triples = None
        self.triples0 = None
        self.positive_graph_train = None
        self.negative_graph_train = None

    def _get_training_set_graph(self, directed=True):
        self.positive_graph_train, self.negative_graph_train = graph.from_edgelist_array_to_graph(self.train_X,
                                                                                                  self.train_y,
                                                                                                  directed=directed)

    def _get_training_triples(self, p0=True, directed=True):
        self._get_training_set_graph(directed)
        if p0:
            self.triples, self.triples0 = graph.get_triples(self.positive_graph_train, self.negative_graph_train, p0)
        else:
            self.triples = graph.get_triples(self.positive_graph_train, self.negative_graph_train, p0)

    def get_training_triples(self, p0=True, directed=True):
        if p0 and self.triples is None and self.triples0 is None:
            self._get_training_triples(p0, directed)
        elif self.triples is None:
            self._get_training_triples(p0, directed)

        if p0:
            return self.triples, self.triples0
        return self.triples

    def get_training_set(self):
        return self.train_X, self.train_y

    def get_testing_set(self):
        return self.test_X, self.test_y

    def get_num_nodes(self):
        return self.id_assigner.count

    def get_shuffled_data(self):
        X_train, y_train = self.get_training_set()
        X_test, y_test = self.get_testing_set()
        X_temp = np.concatenate((X_train, X_test))
        y_temp = np.concatenate((y_train, y_test))
        return X_temp, y_temp



def vote_hamming_distance(votes1, votes2):
    ids = np.array(range(len(votes1)))
    both_voted = ids[(votes1 != 0) & (votes2 != 0)]
    if len(both_voted) == 0:
        return 0
    v1 = votes1[both_voted]
    v2 = votes2[both_voted]
    distance = hamming(v1, v2)
    return distance


class SenateDataset(UnsplitDataset):
    def __init__(self, filepath, class_filepath, delimiter=',', ratio=0.8):
        super().__init__(filepath=filepath, delimiter=delimiter, ratio=ratio)
        self.classes = dict()
        self.nodes = []
        self.labels = []
        with open(class_filepath, 'r') as fp:
            for line in fp:
                id, cl = line.split(delimiter)
                #id = int(id)
                id = self.id_assigner.object2id(id)  
                cl = int(cl)
                self.classes[id] = cl
                self.nodes.append(id)
                self.labels.append(cl)
        self.train_nodes, self.test_nodes, self.train_labels,\
        self.test_labels = train_test_split(self.nodes, self.labels, train_size=ratio,test_size=1-ratio, shuffle=True)



    def get_class(self, node_id):
        return self.classes[node_id]

    def get_node_classes(self, ids=None):
        arr = []
        if ids is None:
            ids = self.classes.keys()
        for id in sorted(ids):
            arr.append(self.get_class(id))
        return np.array(arr)

class RegressionDataset(UnsplitDataset):
    def __init__(self, filepath, class_filepath, delimiter=',', ratio=0.8):
        super().__init__(filepath=filepath, delimiter=delimiter, ratio=ratio)
        self.response = dict()
        self.regression_edges = []
        self.regression_response = []
        self.graph = nx.DiGraph()
        with open(class_filepath, 'r') as fp:
            for line in fp:
                u, v, w = line.split(delimiter)
                u = self.id_assigner.object2id(u)
                v = self.id_assigner.object2id(v)
                w = float(w)
                self.graph.add_edge(u, v, weight=w)

        #         self.response[(u, v)] = w
        #         self.regression_edges.append([u, v])
        #         self.regression_response.append(w)
        # self.regression_edges = np.array(self.regression_edges)
        # self.regression_response = np.array(self.regression_response)
        # self.train_edges, self.test_edges, self.train_response,\
        # self.test_response = train_test_split(self.regression_edges,
        #                                       self.regression_response,
        #                                       train_size=ratio,test_size=1-ratio, shuffle=True)

    def get_response(self, u, v):
        return self.graph[u][v]['weight']




def from_bitcoin_to_regression(filepath, outpath):
    file = open(filepath, 'r')
    outlines = ''
    for line in file:
        u, v, w, t = line.split(',')
        outlines += '{0},{1},{2}\n'.format(u, v, w)
    file.close()

    file = open(outpath, 'w')
    file.write(outlines)
    file.close()


def edgelist_from_vot(filepath, edgelist_path, threshold=0.5):
    file = open(filepath, 'r')
    votes = dict()
    edges = []
    edge_list = ''

    for i, line in enumerate(file):
        voting_record = list(map(int, line.split()))
        voting_record = np.array(voting_record )
        votes[i + 1] = voting_record
    file.close()

    for i in votes.keys():
        for j in votes.keys():
            if i != j:
                i_records = votes[i]
                j_records = votes[j]
                distance = vote_hamming_distance(i_records, j_records)
                sign = -1 if distance < threshold else 1
                edge_list += '{0},{1},{2}\n'.format(i, j, sign)

    file = open(edgelist_path, 'w')
    file.write(edge_list)
    file.close()

def node_props_from_nam(filepath, outpath):
    id_pattern = '([0-9]+)'
    class_pattern = '\(([DRIG])\)'

    out = ''

    id = re.compile(id_pattern)
    class_s = re.compile(class_pattern)

    counter = -1
    assigner = dict()

    file = open(filepath, 'r')
    for line in file:
        line = line.strip()
        if line:
            id_matches = id.search(line)
            class_matches = class_s.search(line)
            if id_matches is None or class_matches is None:
                print('Warning!')
                print(f'Line {line} is bad!')
            id_match = id_matches.group(0)
            class_match = class_matches.group(0)

            if class_match not in assigner:
                counter += 1
                assigner[class_match] = counter
            c_id = assigner[class_match]
            out += '{0},{1}\n'.format(id_match, c_id)
    file.close()

    file = open(outpath, 'w')
    file.write(out)
    file.close()













