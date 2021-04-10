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


def collect_outputs(n, l0, d, prop_mispl, prop_neg, network_no, network_desc,
                         output_descs):
    """This method collects all the outputs indicated by 'output_descs'. Those outputs
    will be later used in the prediction tasks. 
       
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
    :param output_descs: outputs, e.g. consts.OUTPUT_NB_SOLUTIONS, etc. 
    :type output_descs: str list
    """
    outputs = pd.DataFrame([])
    
    network_folder = path.get_input_network_folder_path(n, l0, d, prop_mispl, prop_neg, network_no)
    network_path = os.path.join(network_folder, consts.SIGNED_UNWEIGHTED+".graphml")
    
    # we continue if the corresponding input network exists
    if os.path.exists(network_path):
        eval_partitions_folder_path = path.get_evaluate_partitions_folder_path(n, l0, d, prop_mispl, prop_neg,
                                                     network_no, network_desc)
        #print("..... collecting outputs in "+eval_partitions_folder_path)
        for output_desc in output_descs:
            result = None
            
            if output_desc == consts.OUTPUT_NB_SOLUTIONS or output_desc == consts.OUTPUT_IS_SINGLE_SOLUTION:
                result_filepath = os.path.join(eval_partitions_folder_path,"nb-solution.csv")
                nb_solutions = int(pd.read_csv(result_filepath, index_col=0).values) 
                
                if output_desc == consts.OUTPUT_NB_SOLUTIONS:
                    result = [nb_solutions] # a list of a single value
                if output_desc == consts.OUTPUT_IS_SINGLE_SOLUTION:
                    result = [int(nb_solutions==1)] # a list of a single value
                    
            elif output_desc == consts.OUTPUT_NB_SOLUTION_CLASSES or output_desc == consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES:
                result_filepath = os.path.join(eval_partitions_folder_path,"Best-k-for-kmedoids.csv")
                nb_solution_classes = 1 # by default
                if os.path.exists(result_filepath):
                    nb_solution_classes = int(pd.read_csv(result_filepath, usecols=['Best k for Silhouette']).values) 
                
                if output_desc == consts.OUTPUT_NB_SOLUTION_CLASSES:
                    result = [nb_solution_classes] # a list of a single value
                if output_desc == consts.OUTPUT_IS_SINGLE_SOLUTION_CLASSES:
                    result = [int(nb_solution_classes==1)] # a list of a single value

            elif output_desc == consts.OUTPUT_GRAPH_IMBALANCE_COUNT or output_desc == consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE:
                result_filepath = os.path.join(eval_partitions_folder_path, "imbalance.csv")
                imbalance_count = int(pd.read_csv(result_filepath, usecols=['imbalance count']).values)
                imbalance_percentage = float(pd.read_csv(result_filepath, usecols=['imbalance percentage']).values)
                if output_desc == consts.OUTPUT_GRAPH_IMBALANCE_COUNT:
                    result = [imbalance_count] # a list of a single value
                if output_desc == consts.OUTPUT_GRAPH_IMBALANCE_PERCENTAGE:
                    result = [imbalance_percentage] # a list of a single value
                
            df = pd.DataFrame(data=result, index=consts.COL_NAMES[output_desc]).transpose() # row vector
            outputs = pd.concat([outputs, df], axis=1)

    return outputs       
  
               

def collect_all_outputs(graph_sizes, l0_values, d, prop_mispls, prop_negs, networks, network_desc, output_descs, force=False, verbose=False):
    """This method handles the input signed networks before collecting the indicated outputs.
       
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
    :param output_descs: outputs, e.g. consts.OUTPUT_NB_SOLUTIONS, etc. 
    :type output_descs: str list
    """
    outputs = pd.DataFrame([])
    
    csv_folder_path = path.get_csv_folder_path()
    os.makedirs(csv_folder_path, exist_ok=True)
    result_filename = consts.FILE_CSV_OUTPUTS+".csv"
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

                            if verbose:
                                print(
                                    "... collecting outputs with n="+str(n)+", l0="+str(l0)+", dens="+util.format_4digits(d),
                                    ", propMispl="+util.format_4digits(prop_mispl),
                                    ", propNeg="+util.format_4digits(prop_neg),
                                    ", network="+str(network_no)
                                )
        
                            row = collect_outputs(n, l0, d, prop_mispl, prop_neg, 
                                                 network_no, network_desc, output_descs)
                            if row.size != 0:
                                outputs = outputs.append(row)
                                rownames.append(desc)
        
        outputs.index =  rownames          
        outputs.to_csv(result_filepath, sep=",", quoting=1, index=True)
    else:
        if verbose:
            print(result_filepath+" already exists")
        
