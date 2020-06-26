library(Matrix)
library(igraph)
library(signnet)

setwd("/Users/SUCAL-V/Documents/Université/Stage/Workspace/PycharmProjects/SignedCentrality/tests/")

# Table 5
cat(paste("Table 5\n"))

csv <- read.csv("table_5.csv", header = FALSE)
# csv
m <- as.matrix(csv)
# dim(m)

# Table 5 : undirected
cat(paste("\n\tUndirected :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
adj <- as_adjacency_matrix(g, attr = "sign", sparse = T)
# adj
p <- pn_index(g)
p  # Results are the same as in the article

# Table 5 : in
cat(paste("\n\tIncoming :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
adj <- as_adjacency_matrix(g, "both", attr = "sign", sparse = T)
# adj
p_in <- pn_index(g, mode = "in")
p_in

# Table 5 : out
cat(paste("\n\tOutgoing :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out

cat(paste("\n\n\n"))



# GAMAPOS
cat(paste("GAMAPOS\n"))

csv <- read.csv("GAMAPOS.csv", header = TRUE)
m <- as.matrix(csv)
m <- m[,-c(1)]
# m

# GAMAPOS : undirected
cat(paste("\n\tUndirected :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p

# GAMAPOS : in
cat(paste("\n\tIncoming :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

# GAMAPOS : out
cat(paste("\n\tOutgoing :\n\n"))

g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out

cat(paste("\n\n\n"))



# Sampson Monastery
cat(paste("Sampson Monastery\n"))

csv_directed <- read.csv("sampson_directed.csv", header = FALSE)
csv_undirected <- read.csv("sampson_undirected.csv", header = FALSE)

# Sampson : undirected
cat(paste("\n\tUndirected :\n\n"))

m <- as.matrix(csv_directed)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p

# Sampson : in
cat(paste("\n\tIncoming :\n\n"))

m <- as.matrix(csv_undirected)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

# Sampson : out
cat(paste("\n\tOutgoing :\n\n"))

m <- as.matrix(csv_undirected)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out


# plot(g)


cat(paste("\n\n\n"))











