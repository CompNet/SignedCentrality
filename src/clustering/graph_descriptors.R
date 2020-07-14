#! /usr/bin/Rscript --vanilla
args <- commandArgs(trailingOnly=TRUE)

library(igraph)
library(signnet)
library(XML)

path <- args[1]
res_path <- paste0(path, '/', args[2])
generated_path <- paste0(path, '/', args[3])  # Resource folder to read result files.
r_generated_path <- paste0(path, '/', args[4])  # Resource folder to write result files.
dataset_path <- paste0(path, '/', args[5])
input_dataset_path <- paste0(dataset_path, '/inputs/')
output_dataset_path <- paste0(dataset_path, '/outputs/')
input_files_paths_csv_file <- paste0(args[6])

setwd(path)

if (! dir.exists(r_generated_path)) {
  if (! dir.exists(generated_path)) {
    if (! dir.exists(res_path)) {
      dir.create(res_path)
    }
    dir.create(generated_path)
  }
  dir.create(r_generated_path)
}



# Functions :

source('../functions.R')


export_results <- function(original_path, output_directory_path, results) {
  output_path <- paste0(output_directory_path, )
}

get_graph_from_path <- function (file_name, format = 'graphml') {
  print(paste0(file_name))
  g <- read_graph(file_name, format = format)

  matrix <- get.adjacency(g, attr = 'weight')
  graph <- graph_from_adjacency_matrix(matrix, weighted = "sign", mode = 'undirected')

  return(graph)
}

compute_nodes_number <- function (graph) {
  size <- gsize(graph)

  return(size)
}

compute_negative_ties_ratio <- function (graph) {
  weights <- get.edge.attribute(graph = graph, name = 'sign')
  positive_ties_number <- 0
  negative_ties_number <- 0

  for (weight in weights) {
    if (weight > 0) {
      positive_ties_number <- positive_ties_number + 1
    }
    if (weight < 0) {
      negative_ties_number <- negative_ties_number + 1
    }
  }

  ratio <- (positive_ties_number + negative_ties_number) / negative_ties_number

  return(ratio)
}

compute_signed_triangles <- function (graph) {
  res <- count_signed_triangles(graph)
  res <- c(res['+++'], res['++-'], res['+--'])

  return(res)
}

compute_descriptors <- function (file_name, output_file) {
  graph <- get_graph_from_path(file_name)

  # print(graph)

  nodes_number <- compute_nodes_number(graph)
  negative_ties_ratio <- compute_negative_ties_ratio(graph)
  signed_triangles <- compute_signed_triangles(graph)


}




# Program :


# csv <- read.csv(input_files_paths_csv_file, header = FALSE)
csv <- read.table(input_files_paths_csv_file, header = FALSE, sep = ",", quote = "\"", dec = ".", fill = TRUE, comment.char = "")

csv

for (file_names in csv) {
  for (file_name in file_names) {
    # file_name <- paste0(file_name)

    # print(paste("0 =>\t", file_name[0]))
    print(paste("1 =>\t", file_name[1]))

    # if (file.exists(file_name)) {
    #   print("File exists.")
    #   cat(paste0(file_name))
    #
    #   file <- file(file_name)
    #   open(file, 'r')
    #
    #   if (isOpen(file, rw = 'r')) {
    #     print("File is open.")
    #
    #     graph <- read_graph(file_name, format = 'graphml')
    #     graph
    #     set.graph.attribute(graph, 'sign', get.graph.attribute(graph, 'weight'))
    #     graph
    #   }
    #   else {
    #     print("File isn't open.")
    #   }
    # }
    # else {
    #   print("File doesn't exist.")
    # }

    # compute_descriptors(file_name[0], file_name[1])

  }
}



