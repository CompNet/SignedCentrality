'''
Created on Sep 23, 2020

@author: nejat
'''

from numpy import array
import node_embeddings
import util
import centrality.degree_centrality
import centrality.eigenvector_centrality
import consts
import path
import os
from statistics import mean, stdev 

import pandas as pd

from node_embeddings import sne
from node_embeddings.sne.sne_embedding import SNEEmbedding


def compute_centralities(n, l0, d, prop_mispl, prop_neg, network_no, network_desc, graph_descriptors, force=False, verbose=False):
    """This method computes all the implemented centralities for a set of input 
    parameters.
       
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
    :param graph_descriptors: centralities or embeddings, e.g. consts.CENTR_DEGREE_NEG, consts.CENTR_DEGREE_POS, etc.
    :type graph_descriptors: str list
    """
    network_folder = path.get_input_network_folder_path(n, l0, d, prop_mispl, prop_neg, network_no)
    network_path = os.path.join(network_folder, consts.SIGNED_UNWEIGHTED+".graphml")
    
    # we continue if the corresponding input network exists
    if os.path.exists(network_path):
        g = util.read_graph(network_path, consts.FILE_FORMAT_GRAPHML)
    
        for desc_name in graph_descriptors:
            centr_folder_path = path.get_centrality_folder_path(n, l0, d, prop_mispl, prop_neg,
                                                                 network_no, network_desc)
            if verbose:
                print("computing centrality: "+desc_name+" in "+centr_folder_path)
            os.makedirs(centr_folder_path, exist_ok=True)
        
            result_filename = desc_name+".csv"
            result_filepath = os.path.join(centr_folder_path,result_filename)
            if not os.path.exists(result_filepath) or force:
                result = None

                if desc_name in consts.GRAPH_DESCRIPTORS:
                    # result = consts.GRAPH_DESCRIPTORS[desc_name](g).tolist()
                    result = [v for v in consts.GRAPH_DESCRIPTORS[desc_name](g)]

                # if desc_name == consts.CENTR_DEGREE_NEG:
                #     result = centrality.degree_centrality.NegativeCentrality.undirected(g, False).tolist()
                # elif desc_name == consts.CENTR_DEGREE_POS:
                #     result = centrality.degree_centrality.PositiveCentrality.undirected(g, False).tolist()
                # elif desc_name == consts.CENTR_DEGREE_PN:
                #     result = centrality.degree_centrality.PNCentrality.undirected(g, False).tolist()
                # elif desc_name == consts.CENTR_EIGEN:
                #     result = centrality.eigenvector_centrality.compute_eigenvector_centrality(g)
                # elif desc_name == consts.EMB_SNE:
                #     result = SNEEmbedding.undirected(g)
                #     #print(result)
                    
                # write the centrality values into file (as the number of values as the number of lines)

                ################################
                # TODO: This code shouldn't be here: mean values should be coputed in SRWRCentrality.
                try:  # To avoid problems with SRWRCentrality
                    util.format_4digits(result[0])
                except:
                    # Array size is above 1.
                    result = [(sum(e) / len(e))[0] for e in result]
                ################################

                result_formatted = [util.format_4digits(e) for e in result]
                df = pd.DataFrame({consts.CENT_COL_NAME : result_formatted})
                df.to_csv(result_filepath, sep=",",quoting=1,index=False)
               
                # write the mean of the centrality values
                desc = consts.PREFIX_MEAN+desc_name
                result_filepath = os.path.join(centr_folder_path,consts.PREFIX_MEAN+result_filename)
                result_formatted = util.format_4digits(mean(result))
                df = pd.DataFrame({desc : [result_formatted]})
                df.to_csv(result_filepath, sep=",",quoting=1,index=False)
                    
                # write the standard deviation of the centrality values
                desc = consts.PREFIX_STD+desc_name
                result_filepath = os.path.join(centr_folder_path,consts.PREFIX_STD+result_filename)
                result_formatted = util.format_4digits(stdev(result))
                df = pd.DataFrame({desc : [result_formatted]})
                df.to_csv(result_filepath, sep=",",quoting=1,index=False)
                


def compute_all_centralities(graph_sizes, l0_values, d, prop_mispls, prop_negs, networks, network_desc, graph_descriptors, force=False, verbose=False):
    """This method handles the input signed networks before computing all the 
    implemented centralities.
       
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
    :param graph_descriptors: centralities or embeddings, e.g. consts.CENTR_DEGREE_NEG, consts.CENTR_DEGREE_POS, etc.
    :type graph_descriptors: str list
    """

    for n in graph_sizes:
        for l0 in l0_values:
            for prop_mispl in prop_mispls:
                
                my_prop_negs = prop_negs
                if my_prop_negs is None and d == 1:
                    my_prop_negs = [util.compute_prop_neg(n, l0)]
                    
                for prop_neg in my_prop_negs:
                    for network_no in networks:
                        if verbose:
                            print(
                                "... computing centralities with n="+str(n)+", l0="+str(l0)+", dens="+util.format_4digits(d)
                                , ", propMispl="+util.format_4digits(prop_mispl),
                                ", propNeg="+util.format_4digits(prop_neg)
                                , ", network="+str(network_no)
                            )
    
                        compute_centralities(n, l0, d, prop_mispl, prop_neg, network_no, network_desc, graph_descriptors, force, verbose=verbose)
