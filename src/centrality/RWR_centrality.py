from pyrwr.rwr import RWR


def rwr(g, graph_type):

    rwr = RWR()

    f = open("RWR_temp.txt","w+")

    for i in g.es:
        f.write(i.source, "\t", i.target, "\t", g.es['weight'][i])

    f.close()
    rwr.read_graph("RWR_temp.txt", graph_type)
    r = rwr.compute(seed, c, epsilon, max_iters)
    
