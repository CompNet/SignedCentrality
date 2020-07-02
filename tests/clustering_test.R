#! /usr/bin/Rscript
args <- commandArgs(trailingOnly=TRUE)

library(igraph)
library(signnet)

path <- args[1]
setwd(path)


