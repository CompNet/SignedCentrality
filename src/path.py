'''
Created on Sep 23, 2020

@author: nejat
'''

import os
import consts
import util



def get_input_network_folder_path(n, l0, d, prop_mispl, prop_neg, network_no):
    """This method returns the path of the input network folder
       
    :param n: number of nodes
    :type n: int
    :param l0: number of modules
    :type l0: int
    :param d: density
    :type d: int
    :param prop_mispl
    :type prop_mispl: float
    :param prop_neg
    :type prop_neg: float
    :param network_no
    :type network_no: int
    """
    path = os.path.join(consts.IN_FOLDER, "n="+str(n)+"_l0="+str(l0)+
                        "_dens="+util.format_4digits(d), "propMispl="+util.format_4digits(prop_mispl), 
                        "propNeg="+util.format_4digits(prop_neg), "network="+str(network_no))
    return path



def get_evaluate_partitions_folder_path(n, l0, d, prop_mispl, prop_neg, network_no, network_desc):
    """This method returns the path of the 'evaluate-partitions' folder
       
    :param n: number of nodes
    :type n: int
    :param l0: number of modules
    :type l0: int
    :param d: density
    :type d: int
    :param prop_mispl
    :type prop_mispl: float
    :param prop_neg
    :type prop_neg: float
    :param network_no
    :type network_no: int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    """
    path = os.path.join(consts.EVAL_PARTITIONS_FOLDER, "n="+str(n)+"_l0="+str(l0)+
                        "_dens="+util.format_4digits(d), "propMispl="+util.format_4digits(prop_mispl), 
                        "propNeg="+util.format_4digits(prop_neg), "network="+str(network_no),
                        "ExCC-all", network_desc)
    return path




def get_csv_folder_path():
    """This method returns the path of the csv folder
       
    """
    return consts.CSV_FOLDER




def get_centrality_folder_path(n, l0, d, prop_mispl, prop_neg, network_no, network_desc):
    """This method returns the path of the centrality folder
       
    :param n: number of nodes
    :type n: int
    :param l0: number of modules
    :type l0: int
    :param d: density
    :type d: int
    :param prop_mispl
    :type prop_mispl: float
    :param prop_neg
    :type prop_neg: float
    :param network_no
    :type network_no: int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    :param centr_name
    :type centr_name: str. Some possibilities: CENTR_DEGREE_NEG, CENTR_DEGREE_POS, CENTR_DEGREE_PN, etc.
    """
    path = os.path.join(consts.CENTR_FOLDER, "n="+str(n)+"_l0="+str(l0)+
                        "_dens="+util.format_4digits(d), "propMispl="+util.format_4digits(prop_mispl), 
                        "propNeg="+util.format_4digits(prop_neg), "network="+str(network_no), 
                        network_desc)
    return path



def get_stat_folder_path(n, l0, d, prop_mispl, prop_neg, network_no, network_desc):
    """This method returns the path of the stats network folder
       
    :param n: number of nodes
    :type n: int
    :param l0: number of modules
    :type l0: int
    :param d: density
    :type d: int
    :param prop_mispl
    :type prop_mispl: float
    :param prop_neg
    :type prop_neg: float
    :param network_no
    :type network_no: int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    :param centr_name
    :type centr_name: str. Some possibilities: CENTR_DEGREE_NEG, CENTR_DEGREE_POS, CENTR_DEGREE_PN, etc.
    """
    path = os.path.join(consts.STAT_FOLDER, "n="+str(n)+"_l0="+str(l0)+
                        "_dens="+util.format_4digits(d), "propMispl="+util.format_4digits(prop_mispl), 
                        "propNeg="+util.format_4digits(prop_neg), "network="+str(network_no), 
                        network_desc)
    return path


def get_plot_folder_path(n, l0, d, prop_mispl, prop_neg, network_no, network_desc):
    """This method returns the path of the input network folder
       
    :param n: number of nodes
    :type n: int
    :param l0: number of modules
    :type l0: int
    :param d: density
    :type d: int
    :param prop_mispl
    :type prop_mispl: float
    :param prop_neg
    :type prop_neg: float
    :param network_no
    :type network_no: int
    :param network_desc: network description, i.e. whether network is weighted or unweighted
    :type network_desc: str. One of them: SIGNED_UNWEIGHTED, SIGNED_WEIGHTED
    """
    path = os.path.join(consts.PLOT_FOLDER, "n="+str(n)+"_l0="+str(l0)+
                        "_dens="+util.format_4digits(d), "propMispl="+util.format_4digits(prop_mispl), 
                        "propNeg="+util.format_4digits(prop_neg), "network="+str(network_no),
                        network_desc)
    return path


def get_graphics_folder_path():
    """This method returns the path of the csv folder

    """
    return consts.GRAPHICS_FOLDER
