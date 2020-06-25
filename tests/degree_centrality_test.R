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

#        V1        V2        V3        V4        V5        V6        V7        V8
# 0.9009747 0.8613482 0.9076997 0.8613482 0.8410658 0.8496558 0.8617321 0.9015909
#        V9       V10
# 0.8509848 0.9072930


# Table 5 : in
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

#       V1       V2       V3       V4       V5       V6       V7       V8       V9
# 1.132926 1.260525 1.144659 1.260525 1.179987 1.227421 1.256844 1.131042 1.219903
#      V10
# 1.148475


# Table 5 : out
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out

#       V1       V2       V3       V4       V5       V6       V7       V8       V9
# 1.132926 1.260525 1.144659 1.260525 1.179987 1.227421 1.256844 1.131042 1.219903
#      V10
# 1.148475





# GAMAPOS

csv <- read.csv("GAMAPOS.csv", header = TRUE)
m <- as.matrix(csv)
m = m[,-c(1)]
# m

# GAMAPOS : undirected
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "undirected")
# g
p <- pn_index(g)
p

#    GAVEV    KOTUN      OVE    ALIKA    NAGAM    GAHUK    MASIL    UKUDZ    NOTOH
# 1.111111 1.111111 1.159564 1.079804 1.115326 1.199719 1.272832 1.234552 1.111397
#    KOHIK    GEHAM    ASARO    UHETO    SEUVE    NAGAD     GAMA
# 1.075419 1.162314 1.162314 1.151173 1.075550 1.111111 1.111111


# GAMAPOS : in
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_in <- pn_index(g, mode = "in")
p_in

#     GAVEV     KOTUN       OVE     ALIKA     NAGAM     GAHUK     MASIL     UKUDZ
# 0.9182736 0.9182736 0.9057796 0.9528579 0.9239120 0.8818301 0.8249723 0.8501708
#     NOTOH     KOHIK     GEHAM     ASARO     UHETO     SEUVE     NAGAD      GAMA
# 0.9179550 0.9474681 0.9095186 0.9095186 0.8945808 0.9472861 0.9182736 0.9182736


# GAMAPOS : out
g <- graph_from_adjacency_matrix(m, weighted = "sign", mode = "directed")
# g
p_out <- pn_index(g, mode = "out")
p_out

#     GAVEV     KOTUN       OVE     ALIKA     NAGAM     GAHUK     MASIL     UKUDZ
# 0.9182736 0.9182736 0.9057796 0.9528579 0.9239120 0.8818301 0.8249723 0.8501708
#     NOTOH     KOHIK     GEHAM     ASARO     UHETO     SEUVE     NAGAD      GAMA
# 0.9179550 0.9474681 0.9095186 0.9095186 0.8945808 0.9472861 0.9182736 0.9182736


# plot(g)











