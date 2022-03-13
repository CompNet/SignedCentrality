'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import stats.spectral
import stats.structural
import stats.str_balance

import consts
import path
import os
from statistics import mean, stdev 

import pandas as pd


def compute_stats(n, l0, d, prop_mispl, prop_neg, network_no, network_desc, mystats, force=False, verbose=False):
    """This method computes all the implemented stats for the given signed network.
       
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
    :param stats: graph related statistics, e.g. consts.STATS_SIGNED_TRIANGLES, consts.STATS_POS_NEG_RATIO
    :type stats: str list
    """
    network_folder = path.get_input_network_folder_path(n, l0, d, prop_mispl, prop_neg, network_no)
    network_path = os.path.join(network_folder, consts.SIGNED_UNWEIGHTED+".graphml")
    
    # we continue if the corresponding input network exists
    if os.path.exists(network_path):
        g = util.read_graph(network_path, consts.FILE_FORMAT_GRAPHML)
    
        for stat_name in mystats:
            stat_folder_path = path.get_stat_folder_path(n, l0, d, prop_mispl, prop_neg,
                                                                 network_no, network_desc)
            if verbose:
                print("computing stats: "+stat_name+" in "+stat_folder_path)
            os.makedirs(stat_folder_path, exist_ok=True)
        
            result_filename = stat_name+".csv"
            result_filepath = os.path.join(stat_folder_path,result_filename)
            if not os.path.exists(result_filepath) or force:
                result = None
                colnames = None 
                
                if stat_name == consts.STATS_NB_NODES:
                    result = [g.vcount()]
                    colnames = [consts.COL_NAMES[stat_name]]
                if stat_name == consts.STATS_SIGNED_TRIANGLES:
                    result = stats.str_balance.compute_signed_triangle_ratios(g)
                    result = [util.format_4digits(e) for e in result]
                    colnames = consts.COL_NAMES[stat_name]
                elif stat_name == consts.STATS_LARGEST_EIGENVALUE:
                    result = stats.spectral.retreive_largest_eigenvalue(g)
                    result = [util.format_4digits(result)]
                    colnames = [consts.COL_NAMES[stat_name]]
                elif stat_name == consts.STATS_POS_NEG_RATIO:
                    result = stats.structural.retreive_pos_neg_ratio(g)
                    result = [util.format_4digits(result)]
                    colnames = [consts.COL_NAMES[stat_name]]
                elif stat_name == consts.STATS_POS_PROP:
                    result = stats.structural.retreive_pos_prop(g)
                    result = [util.format_4digits(result)]
                    colnames = [consts.COL_NAMES[stat_name]]
                elif stat_name == consts.STATS_NEG_PROP:
                    result = stats.structural.retreive_neg_prop(g)
                    result = [util.format_4digits(result)]
                    colnames = [consts.COL_NAMES[stat_name]]
                    
                # write the result into file with its column name   
                result_filepath = os.path.join(stat_folder_path,result_filename)
                df = pd.DataFrame(data=result, index=colnames).transpose() # row vector
                df.to_csv(result_filepath, sep=",",quoting=1,index=False)
                
            else:
                if verbose:
                    print("already exists")
                 

def compute_all_stats(graph_sizes, l0_values, d_values, prop_mispls, prop_negs, networks, network_desc, stats, force=False, verbose=False):
    """This method handles the input signed networks before computing all the 
    implemented stats.
       
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
    :param networks: network numbers
    :type networks: a list of int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    :param stats: graph related statistics, e.g. consts.STATS_SIGNED_TRIANGLES, consts.STATS_POS_NEG_RATIO
    :type stats: str list
    """
    
    for d in d_values:
        for n in graph_sizes:
            for l0 in l0_values:
                for prop_mispl in prop_mispls:
                    
                    my_prop_negs = prop_negs
                    if d == 1:
                        my_prop_negs = [util.compute_prop_neg(n, l0)]
                        
                    # print(util.compute_prop_neg(n, l0))
                    
                    for prop_neg in my_prop_negs:
                        for network_no in networks:
                            if verbose:
                                print(
                                    "... computing stats with n="+str(n)+", l0="+str(l0)+", dens="+util.format_4digits(d),
                                    ", propMispl="+util.format_4digits(prop_mispl),
                                    ", propNeg="+util.format_4digits(prop_neg),
                                    ", network="+str(network_no)
                                )
        
                            compute_stats(n, l0, d, prop_mispl, prop_neg, network_no, network_desc, stats, force, verbose=verbose)
