#! /usr/bin/Rscript --vanilla
args <- commandArgs(trailingOnly=TRUE)

library(Matrix)
library(igraph)
library(signnet)

# print(args[1])

path <- args[1]
res_path <- paste0(path, '/res/')  # Resource folder to write result files.
setwd(path)

if (! dir.exists(res_path)) {
  dir.create(res_path)
}

# modes :
mode_undirected <- "undirected"
mode_in <- "in"
mode_out <- "out"

# Functions :

export <- function (values, file_name) {
  path <- paste0(res_path, file_name)

  if (! file.exists(path)) {
    file.create(path)
  }

  write.table(values, file = path, append = FALSE, sep = ',', eol = '\n')
}

compute_centrality <- function (matrix, mode, file_name) {
  cat(paste(file_name, ":"))

  graph_mode <- "undirected"
  pn_mode <- "all"
  if (mode == "in") {
    graph_mode <- "directed"
    pn_mode <- "in"
  }
  else if (mode == "out") {
    graph_mode <- "directed"
    pn_mode <- "out"
  }
  # else, mode == "undirected"

  g <- graph_from_adjacency_matrix(matrix, weighted = "sign", mode = graph_mode)
  adj <- as_adjacency_matrix(g, attr = "sign", sparse = T)
  # adj
  p_in <- pn_index(g, mode = pn_mode)

  # p  # Results are the same as in the article
  export(p, file_name)
}

# Program :


# Table 5
cat(paste("Table 5\n"))

csv <- read.csv("table_5.csv", header = FALSE)
# csv
m <- as.matrix(csv)
# dim(m)

# Table 5 : undirected
# cat(paste("\n\tUndirected :\n\n"))

# g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# # g
# adj <- as_adjacency_matrix(g, attr = "sign", sparse = T)
# # adj
# p <- pn_index(g)
# # p  # Results are the same as in the article
# export(p, '5_undirected.csv')
compute_centrality(m, mode_undirected, '5_undirected.csv')

# Table 5 : in
cat(paste("\n\tIncoming :\n\n"))

# g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# # g
# adj <- as_adjacency_matrix(g, "both", attr = "sign", sparse = T)
# # adj
# p_in <- pn_index(g, mode = "in")
# p_in
# export(p_in, '5_in.csv')
compute_centrality(m, mode_in, '5_in.csv')


# Table 5 : out
cat(paste("\n\tOutgoing :\n\n"))

# g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# # g
# p_out <- pn_index(g, mode = "out")
# p_out
compute_centrality(m, mode_out, '5_out.csv')

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
# adjmatrix <- as_adjacency_matrix(g, attr = "sign")
# adjmatrix

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

m <- as.matrix(csv_undirected)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p

# Sampson : in
cat(paste("\n\tIncoming :\n\n"))

m <- as.matrix(csv_directed)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

# Sampson : out
cat(paste("\n\tOutgoing :\n\n"))

m <- as.matrix(csv_directed)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out



# Gama
cat(paste("GAMA\n"))

csv_directed <- read.csv("gama_directed.csv", header = FALSE)
csv_undirected <- read.csv("gama_undirected.csv", header = FALSE)

# Gama : undirected
cat(paste("\n\tUndirected :\n\n"))

m <- as.matrix(csv_undirected)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p

# Gama : in
cat(paste("\n\tIncoming :\n\n"))

m <- as.matrix(csv_directed)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

# Gama : out
cat(paste("\n\tOutgoing :\n\n"))

m <- as.matrix(csv_directed)
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out


# plot(g)


cat(paste("\n\n\n"))











