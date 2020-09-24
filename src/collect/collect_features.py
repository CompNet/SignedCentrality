'''
Created on Sep 23, 2020

@author: nejat
'''


'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import centrality.degree_centrality
import centrality.eigenvector_centrality
import consts
import path
import os
from statistics import mean, stdev

import pandas as pd


def collect_features(n, l0, d, prop_mispl, prop_neg, network_no, network_desc,
                         centralities, stats):
    """This method collects all the indicated features, which are centrality measures
    and graph-related statistics (number of nodes, etc.).
       
    :param n: int
    :type n: int
    :param l0: number of modules from which the underlying graph is created
    :type l0: int
    :param d: density
    :type d: float
    :param prop_mispl: proportion of misplaced links
    :type prop_mispl: float
    :param prop_neg: proportion of negative links 
    :type prop_neg: float
    :param network_no: network no
    :type network_no: int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    :param centralities: centralities, e.g. consts.CENTR_DEGREE_NEG, consts.CENTR_DEGREE_POS, etc. 
    :type centralities: str list
    :param stats: graph related statistics, e.g. consts.STATS_SIGNED_TRIANGLES, consts.STATS_POS_NEG_RATIO
    :type stats: str list
    """
    features = pd.DataFrame([])
    
    network_folder = path.get_input_network_folder_path(n, l0, d, prop_mispl, prop_neg, network_no)
    network_path = os.path.join(network_folder, consts.SIGNED_UNWEIGHTED+".graphml")
    
    # we continue if the corresponding input network exists
    if os.path.exists(network_path):
        g = util.read_graph(network_path, consts.FILE_FORMAT_GRAPHML)
        
        stats_folder_path = path.get_stat_folder_path(n, l0, d, prop_mispl, prop_neg,
                                                     network_no, network_desc)
        #print("..... collecting features in "+stats_folder_path)
        for stat_name in stats:
            result_filepath = os.path.join(stats_folder_path,stat_name+".csv")
            if os.path.exists(result_filepath):
                df = pd.read_csv(os.path.join(stats_folder_path,stat_name+".csv"), 
                            usecols=consts.COL_NAMES[stat_name])
                features = pd.concat([features, df], axis=1)

        # ===============================================================
        
        cent_folder_path = path.get_centrality_folder_path(n, l0, d, prop_mispl, prop_neg,
                                                             network_no, network_desc)
        #print("..... collecting features in "+cent_folder_path)
        for centr_name in centralities:
            desc = consts.PREFIX_MEAN+centr_name
            result_filepath = os.path.join(cent_folder_path,desc+".csv")
            if os.path.exists(result_filepath):
                df = pd.read_csv(result_filepath, 
                            usecols=[desc])
                features = pd.concat([features, df], axis=1)
                            
            desc = consts.PREFIX_STD+centr_name
            result_filepath = os.path.join(cent_folder_path,desc+".csv")
            if os.path.exists(result_filepath):
                df = pd.read_csv(result_filepath, 
                            usecols=[desc])
                features = pd.concat([features, df], axis=1)

    return features       
                 

def collect_all_features(graph_sizes, l0_values, d, prop_mispls, prop_negs, networks,
                              network_desc, centralities, stats, force=False):
    """This method handles the input singed networks before collecting all the
    indicated features. Those features will be later used in the prediction tasks.
       
    :param graph_sizes: a list of number of nodes
    :type graph_sizes: a list of int
    :param l0_values: number of modules from which the underlying graph is created
    :type l0_values: int list
    :param d: density
    :type d: float
    :param prop_mispls: a list of proportion of misplaced links
    :type prop_mispls: a list of float
    :param prop_negs: a list of proportion of negative links 
    :type prop_negs: a list of float
    :param networks: a list of network no
    :type networks: a list of int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    :param centralities: centralities, e.g. consts.CENTR_DEGREE_NEG, consts.CENTR_DEGREE_POS, etc. 
    :type centralities: str list
    :param stats: graph related statistics, e.g. consts.STATS_SIGNED_TRIANGLES, consts.STATS_POS_NEG_RATIO
    :type stats: str list
    """
    features = pd.DataFrame([])
    
    csv_folder_path = path.get_csv_folder_path()
    os.makedirs(csv_folder_path, exist_ok=True)
    result_filename = consts.FILE_CSV_FEATURES+".csv"
    result_filepath = os.path.join(csv_folder_path,result_filename)
    
    if not os.path.exists(result_filepath) or force:
        rownames = []
        for n in graph_sizes:
            for l0 in l0_values:
                for prop_mispl in prop_mispls:
                    
                    my_prop_negs = prop_negs
                    if my_prop_negs is None and d == 1:
                        my_prop_negs = [util.compute_prop_neg(n, l0)]
                        
                    for prop_neg in my_prop_negs:
                        for network_no in networks:
                            desc = "n="+str(n)+", l0="+str(l0)+", dens="+util.format_4digits(d)+", propMispl="+util.format_4digits(prop_mispl)+", propNeg="+util.format_4digits(prop_neg)+", network="+str(network_no)
                                
                            print("... collecting features with n="+str(n)+", l0="+str(l0)+
                                  ", dens="+util.format_4digits(d), ", propMispl="+
                                  util.format_4digits(prop_mispl), 
                                ", propNeg="+util.format_4digits(prop_neg), 
                                ", network="+str(network_no))
        
                            row = collect_features(n, l0, d, prop_mispl, prop_neg, 
                                                 network_no, network_desc, centralities, stats)
                            if row.size != 0:
                                features = features.append(row)
                                rownames.append(desc)
            
        features.index =  rownames          
        features.to_csv(result_filepath, sep=",", quoting=1, index=True)
    else:
        print(result_filepath+" already exists")        
        
