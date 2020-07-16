#! /usr/bin/Rscript --vanilla
args <- commandArgs(trailingOnly=TRUE)

# args <- c(
#   # "",
#   "../../src/clustering",
#   "../../res/generated/",
#   "../../res/generated/",
#   "../../res/generated/R/",
#   "../../res/clustering_dataset_sample/",
#   "../../res/generated//inputs.xml"
# )

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
input_files_paths_xml_file <- paste0(args[6])

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

write_xml_results <- function (data, new_xml_file_path) {
  root <- newXMLNode('root')
  rows <- dim(data)[1]

  for (i in 1:rows) {
    newXMLNode('descriptor', attrs = c(type = paste(data[i, 1]), value = paste(data[i, 2])), parent = root)
  }

  path_list <- strsplit(new_xml_file_path, '/')[[1]]
  dir_path <- ""

  for (dir in head(path_list, -1)) {
    if(dir_path == "") {
      dir_path <- paste(dir)
    }
    else {
      dir_path <- paste(dir_path, dir, sep = '/')
    }

    if (paste0(dir) == "." || paste0(dir) == "..") {
      next
    }

    if (! dir.exists(dir_path)) {
      dir.create(dir_path)
    }
  }

  if (! file.exists(new_xml_file_path)) {
    file.create(new_xml_file_path)
  }

  # write(root, new_xml_file_path)
  saveXML(root, new_xml_file_path, doctype = "<?xml version='1.0' encoding='utf-8'?>")

  # connection <- file(new_xml_file_path)
  # open(connection, "w+")
  #
  # # write(root, connection)
  # cat(root, file = new_xml_file_path)
  #
  # close(connection)
}

get_graph_from_path <- function (file_name, format = 'graphml') {
  # print(paste0(file_name))
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

  data <- matrix(
    c(
      c("nodes-number", nodes_number),
      c("negative-ties-ratio", negative_ties_ratio),
      c("signed-triangles-ppp", signed_triangles['+++']),
      c("signed-triangles-ppm", signed_triangles['++-']),
      c("signed-triangles-pmm", signed_triangles['+--'])
    ),
    nrow = 5,
    byrow = TRUE
  )

  # for (i in data) {
  #   print(paste0("--------"))
  #   for (j in i) {
  #     print(j)
  #   }
  # }
  #
  # print(paste0("--------"))
  # print("")

  # print(data)

  write_xml_results(data, output_file)
}


# Program :

xml <- xmlParse(input_files_paths_xml_file)
root <- xmlRoot(xml)
nodes <- xmlToList(xml)

# print(nodes)

for (node in nodes) {
  input_file_path <- node['input']
  results_file_path <- node['results']

  compute_descriptors(input_file_path, results_file_path)
}


