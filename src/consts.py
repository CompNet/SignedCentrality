'''
Created on Sep 23, 2020

@author: nejat
'''

import os
from sklearn import metrics

# ===========================
# Path variables
# ===========================
CSV = ".csv"
TXT = ".txt"
PNG = ".png"
PDF = ".pdf"
MAIN_FOLDER = os.path.abspath("..") # the absolute path of the previous level
IN_FOLDER = os.path.join(MAIN_FOLDER, "in")
OUT_FOLDER = os.path.join(MAIN_FOLDER, "out")
EVAL_PARTITIONS_FOLDER = os.path.join(OUT_FOLDER, "evaluate-partitions")
CSV_FOLDER = os.path.join(OUT_FOLDER, "csv")
CENTR_FOLDER = os.path.join(OUT_FOLDER, "centralities")
STAT_FOLDER = os.path.join(OUT_FOLDER, "stats")
PLOT_FOLDER = os.path.join(OUT_FOLDER, "plots")
GRAPHICS_FOLDER = os.path.join(OUT_FOLDER, "graphics")


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

# embeddings
SNE_SAVE_PATH_NAME = "save_path"
SNE_TRAIN_DATA_NAME = "train_data"
SNE_LABEL_DATA_NAME = "label_data"
SNE_WALKS_DATA_NAME = "walks_data"
SNE_EMBEDDING_SIZE_NAME = "embedding_size"
SNE_SAMPLES_TO_TRAIN_NAME = "samples_to_train"
SNE_LEARNING_RATE_NAME = "learning_rate"
SNE_NUM_SAMPLED_NAME = "num_sampled"
SNE_CONTEXT_SIZE_NAME = "context_size"
SNE_BATCH_SIZE_NAME = "batch_size"
SNE_IS_TRAIN_NAME = "is_train"


# Linear regression hyper parameters
class LinearRegression:
    FIT_INTERCEPT = "fit_intercept"
    NORMALIZE = "normalize"
    COPY_X = "copy_X"
    N_JOBS = "n_jobs"
    POSITIVE = "positive"


# SVM hyper parameters
class SVM:
    # Parameters Names
    KERNEL = "kernel"
    GAMMA = "gamma"
    MAX_ITER = "max_iter"
    TOL = "tol"
    SHRINKING = "shrinking"
    PROBABILITY = "probability"
    DECISION_FUNCTION_SHAPE = "decision_function_shape"

    # Parameters Values
    GAMMA_SCALE = "gamma_scale"
    GAMMA_AUTO = "gamma_auto"
    DECISION_FUNCTION_SHAPE_OVO = "ovo"
    DECISION_FUNCTION_SHAPE_OVR = "ovr"


# MLP hyper parameters
class MLP:
    # Parameters Names
    HIDDEN_LAYER_SIZES = "hidden_layer_sizes"
    ACTIVATION = "activation"
    SOLVER = "solver"
    ALPHA = "alpha"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    LEARNING_RATE_INIT = "learning_rate_init"
    POWER_T = "power_t"
    MAX_ITER = "max_iter"
    SHUFFLE = "shuffle"
    RANDOM_STATE = "random_state"
    TOL = "tol"
    VERBOSE = "verbose"
    WARM_START = "warm_start"
    MOMENTUM = "momentum"
    NESTEROVS_MOMENTUM = "nesterovs_momentum"
    EARLY_STOPPING = "early_stopping"
    VALIDATION_FRACTION = "validation_fraction"
    BETA_1 = "beta_1"
    BETA_2 = "beta_2"
    EPSILON = "epsilon"
    N_ITER_NO_CHANGE = "n_iter_no_change"
    MAX_FUN = "max_fun"

    # Parameters Values
    IDENTITY = 'identity'
    LOGISTIC = 'logistic'
    TANH = 'tanh'
    RELU = 'relu'
    LBFGS = 'lbfgs'
    SGD = 'sgd'
    ADAM = 'adam'
    CONSTANT = 'constant'
    INVSCALING = 'invscaling'
    ADAPTIVE = 'adaptive'
    AUTO = 'auto'


# Optimal values for prediction metrics
PREDICTION_METRICS_OPTIMAL_VALUES = {
    metrics.r2_score.__name__: 1,
    metrics.mean_squared_error.__name__: 0,
    metrics.mean_absolute_error.__name__: 0,
}

