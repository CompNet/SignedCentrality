library(Matrix)
library(igraph)
library(signnet)

setwd("/Users/SUCAL-V/Documents/Université/Stage/Workspace/PycharmProjects/SignedCentrality/tests/")

csv <- read.csv("table_5.csv", header = FALSE)

csv

m <- as.matrix(csv)

dim(m)

g <- graph_from_incidence_matrix(m, weighted = TRUE)

g

table(V(g)$type)

p <- pn_index(g)  # TODO : Error here

summary(p)














