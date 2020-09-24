'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import consts
import centrality.degree_centrality
import path

if __name__ == '__main__':
    network_path = "../../in/n=20_l0=3_dens=1.0000/propMispl=0.2000/propNeg=0.7000/network=1/signed-unweighted.graphml"
    g = util.read_graph(network_path, consts.FILE_FORMAT_GRAPHML)
    print(g.ecount()) # get the number of edges
    print(g.vcount()) # get the number of vertices
    result = centrality.degree_centrality.NegativeCentrality.undirected(g, False)
    result_list = result.tolist()
    print(result_list)
    print(len(result_list))