'''
Created on Feb 1, 2022

@author: nejat
'''

import itertools
from time import time

import consts

from descriptors.runner import compute_all_centralities
import stats.runner
import collect.collect_features
import collect.collect_outputs

from prediction import ordinal_variable
from prediction import balanced_data_processing


# =====================================

#GRAPH_SIZES = [20,24]
#L0_VALS = [3]
#PROP_MISPLS = [0.2, 0.3] #[x/20 for x in range(0, 11)] # float range from 0.0 to 1.0 with decimal steps
#DENSITY = 1
#INPUT_NETWORKS = range(1,11)
#PROP_NEGS = None # when density=1, this equals 'None'

GRAPH_SIZES = [16,20,24,28,32,36,40,45,50]
L0_VALS = [2,3,4] #
PROP_MISPLS = [x/20 for x in range(0, 21)] # float range from 0.0 to 1.0 with decimal steps
DENSITY = [0.25, 0.5, 1] # 
INPUT_NETWORKS = range(1,101)
PROP_NEGS = [0.3, 0.5, 0.7] # when density=1, this equals 'None'

# GRAPH_SIZES = [16,20,24,28,32,36,40,45,50] # 
# L0_VALS = [2,3,4] #
# PROP_MISPLS = [x/20 for x in range(0, 21)] # float range from 0.0 to 1.0 with decimal steps
# DENSITY = [0.25, 0.5, 1] # 
# INPUT_NETWORKS = range(1,101)
# PROP_NEGS = [0.3, 0.5, 0.7] # when density=1, this equals 'None'



NETWORK_DESC = consts.SIGNED_UNWEIGHTED

GRAPH_DESCRIPTORS = [
  'PNCentrality', 
  'EigenvectorCentrality', 
  'diversity_coef_centrality', 
  'SNEEmbedding', 
  'SiNEEmbedding', 
  'StEMEmbedding'
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
  consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE
  # In the end of this script, we add two additional 
  # output variables for ordinal classification tasks
]

FORCE = False
#FORCE = True
VERBOSE = True









####################################################################### 
# MAIN
#######################################################################

if __name__ == '__main__':

  program_start_time = time()
  
  print(NETWORK_DESC)
  print(GRAPH_DESCRIPTORS)
  print(STATS)
  print(OUTPUTS)
  
  
  ####################################################################### 
  # PART 1: COMPUTE TWO GROUPS OF FEATURES: 1) CENTRALITY-BASED, 2) NETWORKS STATS
  #######################################################################
   
  compute_all_centralities(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS, INPUT_NETWORKS, NETWORK_DESC, GRAPH_DESCRIPTORS, FORCE, VERBOSE)
  
  stats.runner.compute_all_stats(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS, INPUT_NETWORKS, NETWORK_DESC, STATS, FORCE, VERBOSE)
  
  
  
  ####################################################################### 
  # PART 2: COLLECT FEATURES AND OUTPUTS
  #######################################################################
  
  # ----------------------------------------------------------------------------------------
  # REMARK: In the literature, it has been shown that if a signed graph contains multiple connected components
  #        in its positive graph, then one can solve the CC problem for each component, then combine these solutions for the initial graph.
  #        Therefore, when we collect the processed stats/features/outputs, we discard such graphs for the subsequent prediction analysis.
  #        See the following methods:
  #        --> collect_all_outputs()
  #        --> collect_all_features()
  # ----------------------------------------------------------------------------------------

  collect.collect_features.collect_all_features(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS, INPUT_NETWORKS, NETWORK_DESC, GRAPH_DESCRIPTORS, STATS, FORCE)

  collect.collect_outputs.collect_all_outputs(GRAPH_SIZES, L0_VALS, DENSITY, PROP_MISPLS, PROP_NEGS, INPUT_NETWORKS, NETWORK_DESC, OUTPUTS, FORCE)
  
  
  ####################################################################### 
  # PART 3: POST-PROCESSING ON COLLECTED FEATURES AND OUTPUTS
  #######################################################################
  
  # In the ordinal classification task we create two ordinal output variables
  # based on the number of optimal solutions and the number of solution classes
  ordinal_variable.create_ordinal_output_variables()
  
  # In the classification tasks we need a balanced dataset. However, it is not the case in our dataset.
  # Therefore, we perform the under sampling method to get such balanced datasets, and save them for the sake of reproducibility.
  balanced_data_processing.create_balanced_datasets_with_under_sampling_and_train_split()
    
  program_end_time = time() - program_start_time
  print("Running time of the full program:", program_end_time, "seconds")

