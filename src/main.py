'''
Created on Sep 23, 2020

@author: nejat
'''

import itertools

import consts
import centrality.runner
import stats.runner
import collect.collect_features
import collect.collect_outputs
import prediction.classification
import prediction.regression

import prediction.random_forest_classification

import prediction.feature_ablation
from prediction.hyper_parameters import compare_hyper_parameters

from imblearn.under_sampling import EditedNearestNeighbours

# =====================================
GRAPH_SIZES = [20,24]
L0_VALS = [3]
PROP_MISPLS = [0.2, 0.3] #[x/20 for x in range(0, 11)] # float range from 0.0 to 1.0 with decimal steps
DENSITY = 1
INPUT_NETWORKS = range(1,11)
PROP_NEGS = None # when density=1, this equals 'None'
#PROP_NEGS = [0.3, 0.5, 0.7] # do not uncomment !

NETWORK_DESC = consts.SIGNED_UNWEIGHTED

CENTRALITIES = [
    consts.CENTR_DEGREE_PN, 
    consts.CENTR_EIGEN
]
STATS = [
    consts.STATS_NB_NODES,
    consts.STATS_POS_PROP, 
    consts.STATS_NEG_PROP,
    consts.STATS_POS_NEG_RATIO, 
    consts.STATS_SIGNED_TRIANGLES, 
    consts.STATS_LARGEST_EIGENVALUE
]
OUTPUTS = [
    consts.OUTPUT_NB_SOLUTIONS,
    consts.OUTPUT_IS_SINGLE_SOLUTION,
    consts.OUTPUT_NB_SOLUTION_CLASSES,
    consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES,
    consts.OUTPUT_GRAPH_IMBALANCE_COUNT,
    consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE
]

FORCE = False
# =====================================


if __name__ == '__main__':

    centrality.runner.compute_all_centralities(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS,
                                             INPUT_NETWORKS, NETWORK_DESC, CENTRALITIES, FORCE)

    stats.runner.compute_all_stats(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS,
                                             INPUT_NETWORKS, NETWORK_DESC, STATS, FORCE)

    collect.collect_features.collect_all_features(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS,
                                           INPUT_NETWORKS, NETWORK_DESC, CENTRALITIES, STATS, FORCE)

    collect.collect_outputs.collect_all_outputs(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS,
                                            INPUT_NETWORKS, NETWORK_DESC, OUTPUTS, FORCE)

    features_list = [
        consts.COL_NAMES[consts.STATS_NB_NODES],
        consts.COL_NAMES[consts.STATS_POS_PROP],
        consts.COL_NAMES[consts.STATS_NEG_PROP],
        consts.COL_NAMES[consts.STATS_POS_NEG_RATIO],
        consts.COL_NAMES[consts.STATS_SIGNED_TRIANGLES],
        consts.COL_NAMES[consts.STATS_LARGEST_EIGENVALUE],
        [consts.PREFIX_MEAN+consts.CENTR_DEGREE_PN],
        [consts.PREFIX_STD+consts.CENTR_DEGREE_PN],
        [consts.PREFIX_MEAN+consts.CENTR_EIGEN],
        [consts.PREFIX_STD+consts.CENTR_EIGEN]
    ]
     
    features = list(itertools.chain.from_iterable(features_list))
    print(features)
    output = [consts.OUTPUT_IS_SINGLE_SOLUTION]
    print(output)
    kernel = consts.PREDICTION_KERNEL_LINEAR


    # print(kernel)
    # classification task : one or more solutions
    print("\nSVC :")
    prediction.classification.perform_classification(features, output, kernel, EditedNearestNeighbours(n_neighbors=3))
    print("\nRandom Forest :")
    prediction.random_forest_classification.perform_classification(features, output, 1000)

    # regression task : number of solutions
    output1 = [consts.OUTPUT_NB_SOLUTIONS]
    print("Task:", *output1)
    print("\nSVR :")
    prediction.regression.perform_regression(features, output1, kernel)
    print("\nLinear Regression :")
    prediction.regression.perform_linear_regression(features, output1)
    print("\nMLP Regression :")
    prediction.regression.perform_mlp_regression(features, output1)

    # classification task : one or more classes of solution
    output2 = [consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES]
    print("Task:", *output2)
    print("\nSVC :")
    prediction.classification.perform_classification(features, output2, kernel, EditedNearestNeighbours(n_neighbors=3))
    print("\nRandom Forest :")
    prediction.random_forest_classification.perform_classification(features, output2, 1000)


    # regression task : number of classes of solution
    output3 = [consts.OUTPUT_NB_SOLUTION_CLASSES]
    print("Task:", *output3)
    print("\nSVR :")
    prediction.regression.perform_regression(features, output3, kernel)
    print("\nLinear Regression :")
    prediction.regression.perform_linear_regression(features, output3)
    print("\nMLP Regression :")
    prediction.regression.perform_mlp_regression(features, output3)

    # regression task : graph imbalance
    output4 = [consts.OUTPUT_GRAPH_IMBALANCE_COUNT]
    print("Task:", *output4)
    print("\nSVR :")
    prediction.regression.perform_regression(features, output4, kernel)

    output5 = [consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE]
    print("Task:", *output4)
    print("\nSVR :")
    prediction.regression.perform_regression(features, output5, kernel)

    # feature ablation task
    print("\nTask: feature ablation")
    print("\nSVC :")
    prediction.feature_ablation.feature_ablation_svc_classification(features, output2)
    print("\nRandomForestClassifier :")
    # prediction.feature_ablation.feature_ablation_random_forest_classification(features, output2) # TODO don't uncomment it, with the actual parameters, it could block the computer
    print("\nSVR :")
    prediction.feature_ablation.feature_ablation_svr_regression(features, output1)
    print("\nLinear Regression :")
    prediction.feature_ablation.feature_ablation_linear_regression(features, output1)
    print("\nMLP Regression :")
    prediction.feature_ablation.feature_ablation_mlp_regression(features, output1)  # TODO doesn't work, fix it

    # Hyper-parameters comparison
    print("\nCompare Hyper-Parameters")
    compare_hyper_parameters(features)  # Add outputs here to select comparisons to perform.


    # feature ablation classification tests (TODO this code is to apply feature ablation on specific files already balanced, delete once tests finished)
    import prediction.tmp_feature_ablation

    """print("\nSVC (eq_sol) :")
    prediction.tmp_feature_ablation.feature_ablation_svc_classification_eq_sol(features, output)
    print("\nSVC (eq_solclass) :")
    prediction.tmp_feature_ablation.feature_ablation_svc_classification_eq_solclass(features, output2)"""
    print("\nRandom Forest (eq_sol) :")
    prediction.tmp_feature_ablation.feature_ablation_random_forest_classification_eq_sol(features, output)
    print("\nRandom Forest (eq_solclass) :")
    prediction.tmp_feature_ablation.feature_ablation_random_forest_classification_eq_solclass(features, output2)



