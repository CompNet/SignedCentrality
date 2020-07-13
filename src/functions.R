#! /usr/bin/Rscript --vanilla

library(igraph)
library(signnet)

export <- function (values, file_name) {
  path <- paste0(r_generated_path, file_name)

  if (! file.exists(path)) {
    file.create(path)
  }

  write.table(values, file = path, append = FALSE, sep = ',', eol = '\n')
}

