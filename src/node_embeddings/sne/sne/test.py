import sys
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from random import shuffle
import pickle


def hadamard(x, y):
    return x*y

def l1_weight(x, y):
    return np.absolute(x-y)

def l2_weight(x, y):
    return np.square(x-y)

def concate(x, y):
    return np.concatenate((x, y), axis=1)

def average(x, y):
    return (x+y)/2


def load_model(path):
    with open(path, 'rb') as f:
        emb_vertex = pickle.load(f)
        sign_w = pickle.load(f)
        proj_w = pickle.load(f)
        id2vertex = pickle.load(f)
        vertex2id = pickle.load(f)
        edge_source_id = pickle.load(f)
        edge_target_id = pickle.load(f)
        edge_sign = pickle.load(f)
        node_labels = pickle.load(f)
    return emb_vertex, sign_w, proj_w, id2vertex, vertex2id, edge_source_id, edge_target_id, edge_sign, node_labels


def construct_dataset(edge_sign, id2vertex):
    pos_edges = []
    neg_edges = []
    fake_edges = []
    for edge, sign in edge_sign.items():
        if sign == 1:
            pos_edges.append(edge)
        else:
            neg_edges.append(edge)
    shuffle(pos_edges)
    n = len(neg_edges)
    sub_pos_edges = pos_edges[:n]
    n_nodes = len(id2vertex)
    n_fake = 0
    while True:
        edge = np.random.choice(n_nodes, 2)
        if (edge[0], edge[1]) not in edge_sign:
            fake_edges.append((edge[0], edge[1]))
            n_fake += 1
        if n_fake == n:
            break
    x = sub_pos_edges + neg_edges + fake_edges
    y = [1]*len(sub_pos_edges) + [-1]*len(neg_edges) + [0]*len(fake_edges)
    return x, y


def link_prediction(x, y, emb_vertex, emb_context, op):
    kf = KFold(n_splits=10)
    s_idx, t_idx, signs = [], [], []
    for (s, t), y in zip(x ,y):
        s_idx.append(s)
        t_idx.append(t)
        signs.append(y)
    s_emb = emb_vertex[s_idx]
    t_emb = emb_context[t_idx]
    signs = np.asarray(signs)
    kf_accu = []

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        y_train, y_test = signs[train_index], signs[test_index]
        s_train, s_test = s_emb[train_index], s_emb[test_index]
        t_train, t_test = t_emb[train_index], t_emb[test_index]
        x_train = op(s_train, t_train)
        x_test = op(s_test, t_test)
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(x_train, y_train)
        test_preds = clf.predict(x_test)
        accuracy = np.mean(test_preds == y_test)
        print("Folder:%i, Accuracy: %f" % (i, accuracy))  # PRINT
        kf_accu.append(accuracy)

    print("Average Accuracy: %f" % (np.mean(kf_accu)))


def node_classification(node_labels, emb_vertex, emb_context, id2vertex, op):
    labels = []
    for uid in id2vertex:
        labels.append(node_labels[uid])
    kf = KFold(n_splits=10, shuffle=True)
    kf_accu = []
    labels = np.asarray(labels)

    for i, (train_index, test_index) in enumerate(kf.split(labels)):
        y_train, y_test = labels[train_index], labels[test_index]
        s_train, s_test = emb_vertex[train_index], emb_vertex[test_index]
        t_train, t_test = emb_context[train_index], emb_context[test_index]
        x_train = op(s_train, t_train)
        x_test = op(s_test, t_test)
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        test_preds = clf.predict(x_test)
        accuracy = clf.score(x_test, y_test)
        print("Folder:%i, Accuracy: %f" % (i, accuracy))  # PRINT
        print(classification_report(y_test, test_preds))  # PRINT
        kf_accu.append(accuracy)

    print("Average Accuracy: %f" % (np.mean(kf_accu)))
    # Average Accuracy: 0.823590


if __name__ == '__main__':

    emb_vertex, sign_w, proj_w, id2vertex, vertex2id, edge_source_id, edge_target_id, edge_sign, node_labels = \
        load_model('lbl_wiki_edit_emb.pkl')

    print("=========== Node Classification ===========")

    node_classification(emb_vertex=emb_vertex, emb_context=proj_w, node_labels=node_labels, id2vertex=id2vertex, op=concate)

    print("============= Link Prediction =============")

    link_x, link_y = construct_dataset(edge_sign=edge_sign, id2vertex=id2vertex)

    print("\n", "     ------ Operator : Hadamard ------     ", sep="")
    link_prediction(x=link_x, y=link_y, emb_vertex=emb_vertex, emb_context=proj_w, op=hadamard)

    print("\n", "     ------ Operator : L1 Weight -----     ", sep="")
    link_prediction(x=link_x, y=link_y, emb_vertex=emb_vertex, emb_context=proj_w, op=l1_weight)

    print("\n", "     ------ Operator : L2 Weight -----     ", sep="")
    link_prediction(x=link_x, y=link_y, emb_vertex=emb_vertex, emb_context=proj_w, op=l2_weight)

    print("\n", "     ------- Operator : Average ------     ", sep="")
    link_prediction(x=link_x, y=link_y, emb_vertex=emb_vertex, emb_context=proj_w, op=average)








