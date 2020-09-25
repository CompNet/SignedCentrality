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
    consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES
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
    print(kernel)
    prediction.classification.perform_classification(features, output, kernel)
    
