#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions related to the measure of A simple approach for quantifying node centrality
in signed and directed social networks centrality.

The measure is computed by following the method of Wei-Chung Liu, Liang-Cheng Huang, Chester Wai-Jen Liu & Ferenc Jordán.

I will use the rpy2 package to execute the R code implemented by the authors directly in this Python module.
Documentation for this package can be found here : https://rpy2.github.io/doc/latest/html/index.html

.. note: WC. Liu, LC. Huang, C. WJ. Liu et J. Ferenc. «A simple approach for quantifying node centrality
in signed and directed social networks». In :Applied Network Science5.1 (août 2020), p. 46. issn : 2364-8228.
doi :10.1007/s41109-020-00288-w

"""

from descriptors import GraphDescriptor
import util
import rpy2.robjects as robjects


class NodeEffect(GraphDescriptor):
    """
    This class is used to compute node effects centralities
    """

    global global_resu

    @staticmethod
    def perform_all(graph, **kwargs):
        """
        Compute this centrality.
        """
        # todo ajouter code centralité
        adjacence_matrix = util.get_matrix(graph).toarray() # ndarray that contains all links for all nodes (as an adjacence list)
        nr, nc = adjacence_matrix.shape # obtaining the number of row and the number of column of the ndarray object
        R_adjacence_matrix = robjects.r.matrix(adjacence_matrix, nrow=nr, ncol=nc) # https://stackoverflow.com/questions/27283381/how-to-convert-numpy-array-to-r-matrix
        # robjects.globalenv['matrix_data'] = R_adjacence_matrix # https://stackoverflow.com/questions/30525027/passing-a-python-variable-to-r-using-rpy2
        robjects.r.assign("data", R_adjacence_matrix) # the R adjacence matrix created above is now associated as a R variable named "data"

        robjects.r('''data
        # Data must be of the edge list format:
        # A B 1 means A influences B positively;
        # C D 1 means C influences D positively;
        # A C -1 means A influences C negatively etc.
        
        numstep<-2 # This sets the number of steps used, and in this version, step lengths have equal weights.
        
        nodeID<-levels(factor(c(as.character(data[,1]),as.character(data[,2])))) # obtention du noms des noeuds
        numnode<-length(nodeID) # obtention du nombre de noeuds
        mx<-PosEf<-NegEf<-TotalEf<-NetEf<-PosCI<-NegCI<-NowPos<-NowNeg<-matrix(rep(0,numnode^2),nrow=numnode,ncol=numnode) # créations de plusieurs matrices de données, avec comme nombre de lignes et de colonnes numnode
        rownames(mx)<-rownames(PosEf)<-rownames(NegEf)<-rownames(TotalEf)<-rownames(NetEf)<-rownames(PosCI)<-rownames(NegCI)<-rownames(NowPos)<-rownames(NowNeg)<-nodeID # application des noms des noeuds sur les lignes
        colnames(mx)<-colnames(PosEf)<-colnames(NegEf)<-colnames(TotalEf)<-colnames(NetEf)<-colnames(PosCI)<-colnames(NegCI)<-colnames(NowPos)<-colnames(NowNeg)<-nodeID # application des noms des noeuds sur les colonnes
        for (i in 1:length(data[,1])){
        mx[as.character(data[i,1]),as.character(data[i,2])]<-data[i,3]} # mx est la matrice d'adjacence
        mx
        
        for (i in 1:numnode)
        {for (j in 1:numnode) 
        {if (mx[j,i]>0) PosEf[j,i]<-mx[j,i]/sum(abs(mx[,i])) # remplissage de la matrice des effets positifs (note : abs() permet d'obtenir la valeur absolue)
        if (mx[j,i]<0) NegEf[j,i]<-mx[j,i]/sum(abs(mx[,i]))}} # remplissage de la matrice des effets négatifs
        
        for (i in 1:numstep)
        {
        if (i==1) # étape 1
        {PosCI<-PosEf # copie de la matrice des effets positifs dans la matrice des "effets positifs indirects cumulés" (?)
        NegCI<-NegEf # copie de la matrice des effets négatifs dans la matrice des "effets négatifs indirects cumulés" (?)
        TotalEf<-TotalEf+PosCI+abs(NegCI) # remplissage de la matrice des effets totaux cumulés par addition de la matrice des effets négatifs indirects et de la matrice des effets positifs indirects. Les valeurs dans cette matrices sont positives, du fait de l'utilisation de abs()
        NetEf<-NetEf+PosCI+NegCI} # remplissage de la metrice des net effect par addition des mêmes matrices que ci-dessus
        else # étape 2 et ultérieur
        {NowPos<-PosCI%*%PosEf+NegCI%*%NegEf # remplissage de la matrice temporaire de valeurs positive par multiplication de matrices des effets positifs avec la matrice des effets positifs indirects cumulés; et vice-versa; puis addition des 2 matrices résultantes
        NowNeg<-PosCI%*%NegEf+NegCI%*%PosEf # remplissage de la matrice temporaire de valeurs négative par multiplication de matrices des effets négatifs avec la matrice des effets positifs indirects cumulés; et vice-versa; puis addition des 2 matrices résultantes
        TotalEf<-TotalEf+NowPos+abs(NowNeg) # modification de la matrice des effets totaux en utilisant cette fois les matrices temporaires calculées ci-dessus. Les valeurs dans cette matrices sont positives, du fait de l'utilisation de abs()
        NetEf<-NetEf+NowPos+NowNeg # modification de la matrice des effets totaux en utilisant cette fois les matrices temporaires calculées ci-dessus
        PosCI<-NowPos # la "matrice temporaire" est copiée dans la matrice globale
        NegCI<-NowNeg} # la "matrice temporaire" est copiée dans la matrice globale
        }
        
        TotalIndex<-numeric(numnode) # crée une variable de type num de taille "numnode" pour avoir l'effet total pour un noeud i
        NetIndex<-numeric(numnode) # idem pour avoir le net effect exercé par le noeud i
        NetIndex1<-numeric(numnode) # idem pour avoir le net effet reçu par le noeud i
        for (i in 1:numnode) TotalIndex[i]<-sum(TotalEf[i,]) # Total effect for node i.
        for (i in 1:numnode) NetIndex[i]<-sum(NetEf[i,]) # Net effect exerted for node i. 
        for (i in 1:numnode) NetIndex1[i]<-sum(NetEf[,i]) # Net effect received for node i.
        
        # Results are in a dataframe called resu. 
        # The 1st column is the node ID, 2nd column is the total effect, 
        # 3rd column is the net effect exerted, and the 4th column is the net effect received. 
        
        resu<-data.frame(nodeID,TotalIndex,NetIndex,NetIndex1)
        resu''')

        global_resu = robjects.globalenv['resu']
        print("resultat : "+global_resu)

        total_total_effect = NodeEffectTotalIndex.perform(graph, **kwargs)
        total_net_effect_exerted = NodeEffectNetIndex.perform(graph, **kwargs)
        total_net_effect_received = NodeEffectNetIndex1.perform(graph, **kwargs)

        return total_total_effect, total_net_effect_exerted, total_net_effect_received


class NodeEffectTotalIndex(NodeEffect):
    """
    This class will return the value of the total effect of a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return total effect value
        """
        robjects.globalenv['resu'] = global_resu
        robjects.r('''totalTotalIndex<-sum(resu$TotalIndex) # somme de tous les effets totaux ''')
        total_total_effect = robjects.globalenv['totalTotalIndex']
        return total_total_effect


class NodeEffectNetIndex(NodeEffect):
    """
    This class will return the value of the net effect exerted by a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return net effect exerted value
        """
        robjects.globalenv['resu'] = global_resu
        robjects.r('''totalNetIndex<-sum(resu$NetIndex) # somme de tous les net effect exercés''')
        total_net_effect_exerted = robjects.globalenv['totalNetIndex']
        return total_net_effect_exerted


class NodeEffectNetIndex1(NodeEffect):
    """
    This class will return the value of the net effect received by a node
    """

    @staticmethod
    def perform(graph, **kwargs):
        """
        return net effet received value
        """
        robjects.globalenv['resu'] = global_resu
        robjects.r('''totalNetIndex1<-sum(resu$NetIndex1) # somme de tous les net effect reçus''')
        total_net_effect_received = robjects.globalenv['totalNetIndex1']
        return total_net_effect_received
