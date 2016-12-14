import sys
import preprocessing as prep
import pandas as pd
import numpy as np
import networkx as nx

from bb_binary import FrameContainer, Repository, load_frame_container
from pandas import DataFrame, Series

# c = float(sys.argv[1])
# d = int(sys.argv[2])
# m = sys.argv[3]
# l = int(sys.argv[4])

def get_files(path):
	repo = Repository(path)
	file = list(repo.iter_fnames())
	a = [f.split('/')[-1].split("_")[1] for f in file]
	l = len(a)/4
	npa = np.array(file).reshape(int(l),4)
	return npa

files = get_files(path)

interactions = Series()

for file_list in files:

	dataframes = np.empty(4, dtype=object)

	for i in list(range(4)):
		fc = load_frame_container(file_list[i])
		df = prep.get_dataframe(fc)
		df = prep.calcIds(df,CONFIDENCE)

		camIdx = int(file_list[i].split("/")[-1].split("_")[1])
		dataframes[camIdx] = df

	# cam 0 und cam1 nach rechts verschieben
	dataframes[0].xpos = dataframes[0].xpos + xmax
	dataframes[1].xpos = dataframes[1].xpos + xmax

	# Seiten zusammenfugen
	side0 = pd.concat([dataframes[3], dataframes[0]])
	side1 = pd.concat([dataframes[2], dataframes[1]])

	close1 = prep.get_close_bees(side0, DISTANCE)
	close2 = prep.get_close_bees(side1, DISTANCE)

	close = pd.concat([close1,close2])

	p = prep.bee_pairs_to_timeseries(close)

	edges = prep.extract_interactions(p,LENGTH)

	interactions = pd.concat([interactions, edges])

G = prep.create_graph2(interactions)
print(nx.info(G))

nx.write_graphml(G, filename + ".graphml")