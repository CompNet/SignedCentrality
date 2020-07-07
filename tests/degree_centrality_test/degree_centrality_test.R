#! /usr/bin/Rscript --vanilla
args <- commandArgs(trailingOnly=TRUE)

library(Matrix)
library(igraph)
library(signnet)
library(tools)

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
  cat(paste(file_name, "\n"))

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
  p <- pn_index(g, mode = pn_mode)

  export(p, file_name)
}

compute_centralities <- function (matrix, name) {
  undirected_name <- paste0(name, '_undirected.csv')
  in_name <- paste0(name, '_in.csv')
  out_name <- paste0(name, '_out.csv')

  mode_undirected <- "undirected"
  mode_in <- "in"
  mode_out <- "out"

  compute_centrality(matrix, mode_undirected, undirected_name)
  compute_centrality(matrix, mode_in, in_name)
  compute_centrality(matrix, mode_out, out_name)
}

compute_centralities_from_csv <- function (csv, name) {
  m <- as.matrix(csv)
  compute_centralities(m, file_path_sans_ext(basename(name)))
}

compute_centralities_from_csv_file <- function (csv_path, header = FALSE) {
  csv <- read.csv(csv_path, header = header)
  m <- as.matrix(csv)
  if (header == TRUE) {
    m <- m[,-c(1)]
  }

  compute_centralities(m, file_path_sans_ext(basename(csv_path)))
}

# Program :


# Table 5
cat(paste("Table 5\n"))

compute_centralities_from_csv_file('table_5.csv', header = FALSE)



# GAMAPOS
cat(paste("GAMAPOS\n"))

compute_centralities_from_csv_file("GAMAPOS.csv", header = TRUE)



# Sampson Monastery
cat(paste("Sampson Monastery\n"))

csv_directed <- read.csv("sampson_directed.csv", header = FALSE)
csv_undirected <- read.csv("sampson_undirected.csv", header = FALSE)

compute_centralities_from_csv(csv_directed, 'sampson_directed')
compute_centralities_from_csv(csv_undirected, 'sampson_undirected')


# Gama
cat(paste("GAMA\n"))

csv_directed <- read.csv("gama_directed.csv", header = FALSE)
csv_undirected <- read.csv("gama_undirected.csv", header = FALSE)

compute_centralities_from_csv(csv_directed, 'gama_directed')
compute_centralities_from_csv(csv_undirected, 'gama_undirected')










