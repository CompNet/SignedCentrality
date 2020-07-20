#! /usr/bin/env Rscript --vanilla
args <- commandArgs(trailingOnly=TRUE)

library(igraph)
library(signnet)

path <- args[1]
res_path <- paste0(path, '/res/')
generated_path <- paste0(path, '/res/generated/')  # Resource folder to read result files.
r_generated_path <- paste0(path, '/res/generated/R/')  # Resource folder to write result files.
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

# modes :
mode_undirected <- "undirected"
mode_in <- "in"
mode_out <- "out"

# Functions :

source('../../src/functions.R')

# Program :

