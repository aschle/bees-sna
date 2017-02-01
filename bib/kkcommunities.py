import networkx as nx
import preprocessing as prep
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import json
import os
import datetime
import pytz
import matplotlib.pyplot as plt
from collections import Counter
from bb_binary import load_frame_container, Repository
from pandas import Series, DataFrame
import sqlite3
import igraph as ig
import pickle
import sys

def removeEdges(G, limit):
    for e in G.edges():
        if (G.get_edge_data(e[0],e[1]).get("weight") < limit):
            G.remove_edge(e[0],e[1])
            
    
    Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    print("Number of components: {}".format(len(Gcc)))

    size_components = []
    for comp in Gcc:
        size_components.append(nx.number_of_nodes(comp))

    print(Counter(list(size_components)))
    
    return Gcc[0]

def run(filename, edge_th, kcc_const, save_as):

	G = nx.read_graphml(filename)
	print(nx.info(G))
	Ge = removeEdges(G, edge_th)
	print(nx.info(Ge))
	c = list(nx.community.k_clique_communities(Ge,kcc_const))
	print(c)
	pickle.dump( c, open( "{}.p".format(save_as), "wb" ) )


if __name__ == '__main__':

	if (len(sys.argv) == 5 ):
		path = sys.argv[1]		
		e = int(sys.argv[2])
		c = int(sys.argv[3])
		f = str(sys.argv[4])

		run(path, e, c, f)
	else:
		print("Usage:\npython3 pipeline.py <path> <edge th> <clique size> <filename>")
		print("Example:\npython3 blabla.py 'networks-days/2015-08-21T00:00:00Z_1d_allCams_0.99conf_160dist_6ilen.graphml' 2 3 myfilename")


# networks-days/2015-08-21T00:00:00Z_1d_allCams_0.99conf_160dist_6ilen.graphml
# 2
# 3
# save_name