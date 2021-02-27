from pyrwr.rwr import RWR


def perform_rwr(graph, graph_type):
    '''
    This method performs the srwr algorithm to calculate the centrality values for
    the nodes of a given graph.

    :param graph: i-graph object
    :type graph: i-graph object
    :param graph_type: indicates the type of the graph (undirected or directed)
    :type graph_type: string
    '''

    rwr = RWR()

    f = open("RWR_temp.txt","w+")

    for i in graph.es:
        f.write(i.source, "\t", i.target, "\t", graph.es['weight'][i])

    f.close()
    rwr.read_graph("RWR_temp.txt", graph_type)
    r = rwr.compute(seed, c, epsilon, max_iters)

    rwr = []
    for value in r:
        rwr.append(r)
        
    graph.vs['rwr'] = rwr
    return graph
