library(Matrix)
library(igraph)
library(signnet)

setwd("/Users/SUCAL-V/Documents/Université/Stage/Workspace/PycharmProjects/SignedCentrality/tests/")

# Table 5

csv <- read.csv("table_5.csv", header = FALSE)
# csv
m <- as.matrix(csv)
# dim(m)

# Table 5 : undirected
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p  # Results are the same as in the article

# Table 5 : in
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

# Table 5 : out
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out

# plot(g)











