'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import scipy.special


def compute_signed_triangle_ratios(g):
    """This method computes tree types of signed triangles of a given signed graph:
        - PPP: positive-positive-positive triangle
        - PPN: positive-positive-negative triangle
        - PNN: positive-negative-negative triangle
    
    Note that if we change the order of those triangles or something else,
    we need to ensure that the corresponding column name(s) in consts.COL_NAME (in consts.py) is still OK.
       
    :param g: graph
    :type g: igraph.Graph
    :return: the ratio of three signed triangles: PPP, PPN and PNN
    :rtype: float list of size 3
    """
    ppp_count = 0
    ppn_count = 0
    pnn_count = 0
    
    n = g.vcount()
    total_count = scipy.special.comb(n, 3)
    
    adj_matrix = util.get_matrix(g)
    n = g.vcount()
    for v1 in range(0,n-2):
        for v2 in range(v1+1,n-1):
            for v3 in range(v2+1,n):
                p_count = 0
                if adj_matrix[v1, v2]>0:
                    p_count += 1
                if adj_matrix[v1, v3]>0:
                    p_count += 1
                if adj_matrix[v2, v3]>0:
                    p_count += 1
                    
                if p_count==1:
                    pnn_count += 1
                if p_count==2:
                    ppn_count += 1
                if p_count==3:
                    ppp_count += 1
    
    ppp_ratio = ppp_count/total_count
    ppn_ratio = ppn_count/total_count
    pnn_ratio = pnn_count/total_count
    
    ratios = [ppp_ratio, ppn_ratio, pnn_ratio]
    return ratios

