
import numpy as np
import sklearn.metrics as metrics
from imblearn.under_sampling import RandomUnderSampler


def measure_quality(y, predictions, probs):
    macro_f1 = metrics.f1_score(y, predictions, average='macro')
    micro_f1 = metrics.f1_score(y, predictions, average='micro')
    #average_percision_score = metrics.average_precision_score(y, probs)
    #auc = metrics.roc_auc_score(y, probs)
    #kappa = metrics.cohen_kappa_score(y, predictions)
    #mathew = metrics.matthews_corrcoef(y, predictions)
    #confustion_matrix = metrics.confusion_matrix(y, predictions)
    #classification_report = metrics.classification_report(y, predictions)

    metric_reports = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        #'average_percision_score':average_percision_score,
        #'auc': auc,
        #'confusion_matrix': confustion_matrix,
        #'classification_report': classification_report,
        #'kappa': kappa,
        #'mathew': mathew
    }

    return metric_reports



def train_and_evaluate_node_classifier(clf, model, node_train, y_train, node_test, y_test):
    
    # train classifier for nodes
    X_train = []
    for node in node_train:
        x = model.get_embedding(node)
        X_train.append(x)
    X_train = np.array(X_train)
    clf.fit(X_train, y_train)

    # test classifier's competencies 

    # assemble train set
    X_test = []
    for node in node_test:
        x = model.get_embedding(node)
        X_test.append(x)
    X_test = np.array(X_test)

    predictions = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]

    reports = measure_quality(y_test, predictions, probs)
    return reports

    






def train_and_evaluate_classifier(clf, model, X_train, y_train, X_test, y_test,
                                  operation='hadamard', undersample=True, ratio=0.8,
                                  print_progress=True):
    num_negative = len(y_test[y_test == 0])
    num_positive = len(y_test[y_test == 1])
    positive_to_sample = min(int(num_negative * ratio), num_positive)
    ratio_dict = {0: num_negative, 1: positive_to_sample}
    rus = RandomUnderSampler(return_indices=False, ratio=ratio_dict)
    X_sampled, y_sampled = X_train, y_train
    if undersample:
        X_sampled, y_sampled = rus.fit_sample(X_train, y_train)
    if print_progress:
        print('Assembling training set features....')
    X = []
    for u, v in X_sampled:
        X.append(model.get_edge_features(u, v, operation))
    y = y_sampled
    X = np.array(X)

    if print_progress:
        print('Fitting classifier model')

    clf.fit(X, y)


    if print_progress:
        print('Assembling testing set features')

    X_sampled, y_sampled = X_test, y_test
    #X_sampled, y_sampled = rus.fit_sample(X_test, y_test)
    X = []
    for u, v in X_sampled:
        X.append(model.get_edge_features(u, v, operation))
    y = y_sampled
    X = np.array(X)
    predictions = clf.predict(X)
    probs = clf.predict_proba(X)[:,1]

    macro_f1 = metrics.f1_score(y, predictions, average='macro')
    micro_f1 = metrics.f1_score(y, predictions, average='micro')
    average_percision_score = metrics.average_precision_score(y, probs)
    auc = metrics.roc_auc_score(y, probs)
    kappa = metrics.cohen_kappa_score(y, predictions)
    mathew = metrics.matthews_corrcoef(y, predictions)
    confustion_matrix = metrics.confusion_matrix(y, predictions)
    classification_report = metrics.classification_report(y, predictions)

    metric_reports = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'average_percision_score':average_percision_score,
        'auc': auc,
        'confusion_matrix': confustion_matrix,
        'classification_report': classification_report,
        'kappa': kappa,
        'mathew': mathew
    }

    return metric_reports

