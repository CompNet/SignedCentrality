'''
Created on Sep 23, 2020

@author: nejat
'''

import os

# ===========================
# Path variables
# ===========================
MAIN_FOLDER = os.path.abspath("..") # the absolute path of the previous level
IN_FOLDER = os.path.join(MAIN_FOLDER, "in")
OUT_FOLDER = os.path.join(MAIN_FOLDER, "out")
EVAL_PARTITIONS_FOLDER = os.path.join(OUT_FOLDER, "evaluate-partitions")
CSV_FOLDER = os.path.join(OUT_FOLDER, "csv")
CENTR_FOLDER = os.path.join(OUT_FOLDER, "centralities")
STAT_FOLDER = os.path.join(OUT_FOLDER, "stats")
PLOT_FOLDER = os.path.join(OUT_FOLDER, "plots")


# ===========================
# Other Variables
# ===========================
# graph nature related
SIGNED_WEIGHTED = "signed-weighted"
SIGNED_UNWEIGHTED = "signed-unweighted"

# graph edge related
EDGE_WEIGHT_ATTR = "weight"
EDGE_SIGN_ATTR = "sign"

# graph file formats
FILE_FORMAT_CSV = "csv"
FILE_FORMAT_GRAPHML = "graphml"

# file prefixes
PREFIX_MEAN = "Mean_"
PREFIX_STD = "Std_"

# some filenames
FILE_CSV_OUTPUTS = "outputs"
FILE_CSV_FEATURES = "features"
FILE_CSV_PREDICTED_VALUES = "predicted_values"


# outputs
OUTPUT_NB_SOLUTIONS = "nb_solutions"
OUTPUT_IS_SINGLE_SOLUTION = "single_solution"
OUTPUT_NB_SOLUTION_CLASSES = "nb_solution_classes"
OUTPUT_IS_SINGLE_SOLUTION_CLASSES = "single_solution_class"
OUTPUT_GRAPH_IMBALANCE_COUNT = "imbalance_count"
OUTPUT_GRAPH_IMBALANCE_PERCENTAGE = "imbalance_percentage"


# centralities
CENTR_DEGREE_NEG = "degree_neg"
CENTR_DEGREE_POS = "degree_pos"
CENTR_DEGREE_PN = "degree_pn"
CENTR_EIGEN = "eigen"

# stats
STATS_NB_NODES = "nb_nodes"
STATS_POS_NEG_RATIO = "pos_neg_ratio"
STATS_POS_PROP = "pos_prop"
STATS_NEG_PROP = "neg_prop"
STATS_SIGNED_TRIANGLES = "signed_triangles"
STATS_LARGEST_EIGENVALUE = "largest_eigenvalue"

# column names
CENT_COL_NAME = "centrality"
MEAN_COL_NAME = "Mean"
STD_COL_NAME = "Std"

COL_NAMES = {
    # features
    STATS_LARGEST_EIGENVALUE : [STATS_LARGEST_EIGENVALUE], 
    STATS_SIGNED_TRIANGLES : ["ppp","ppn","pnn"],
    STATS_POS_NEG_RATIO : [STATS_POS_NEG_RATIO],
    STATS_POS_PROP : [STATS_POS_PROP],
    STATS_NEG_PROP : [STATS_NEG_PROP],
    STATS_NB_NODES : [STATS_NB_NODES],
    # outputs
    OUTPUT_NB_SOLUTIONS : [OUTPUT_NB_SOLUTIONS],
    OUTPUT_IS_SINGLE_SOLUTION : [OUTPUT_IS_SINGLE_SOLUTION],
    OUTPUT_NB_SOLUTION_CLASSES : [OUTPUT_NB_SOLUTION_CLASSES],
    OUTPUT_IS_SINGLE_SOLUTION_CLASSES : [OUTPUT_IS_SINGLE_SOLUTION_CLASSES],
    OUTPUT_GRAPH_IMBALANCE_COUNT : [OUTPUT_GRAPH_IMBALANCE_COUNT],
    OUTPUT_GRAPH_IMBALANCE_PERCENTAGE : [OUTPUT_GRAPH_IMBALANCE_PERCENTAGE]
}

# classification
PREDICTION_KERNEL_LINEAR = "linear"
PREDICTION_KERNEL_POLY = "poly"
PREDICTION_KERNEL_RBF = "rbf"
PREDICTION_KERNEL_SIGMOID = "sigmoid"


