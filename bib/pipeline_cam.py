import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import preprocessing as prep
import sys

from multiprocessing import Pool
from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple 
from pandas import DataFrame, Series

def get_files(path, camid):
	repo = Repository(path)
	files = list(repo.iter_fnames(cam=camid))
	return files

def generate_network(index, file, confidence, distance, ilen):
    
	fc = load_frame_container(file)
	df = prep.get_dataframe2(fc)
	df = prep.calcIds(df,confidence)
	close = prep.get_close_bees_ckd(df, distance)
	p = prep.bee_pairs_to_timeseries(close)

	return prep.extract_interactions(p,ilen)


def run(p, conf=.95, dist=160, ilen=3, c=None, filename="template", camid=0):
	path = p
	confidence = conf
	distance = dist
	interaction_len = ilen
	cpus = c

	# make file file packges
	files = get_files(path, camid)

	# how many cpus
	if (cpus == None):
		max_num_cpu = multiprocessing.cpu_count()
		cpus = max_num_cpu - (max_num_cpu/2/2)

	pool = multiprocessing.Pool(cpus)

	tasks = []

	for e, f in enumerate(files):
		tasks.append((e, f, confidence, distance, interaction_len))

	results = [pool.apply_async( generate_network, t ) for t in tasks]

	# Egebnisse zusammenbasteln
	edges = []

	for result in results:
		edges.append(result.get())

	G = prep.create_graph2(pd.concat(edges))
	nx.write_graphml(G, "{}_{}conf_{}dist_{}ilen_cam{}.graphml".format(
		filename,
		str(confidence),
		str(distance),
		str(interaction_len),
		str(camid)))

	print(nx.info(G))


if __name__ == '__main__':

	if (len(sys.argv) == 8 ):
		path = sys.argv[1]		
		conf = float(sys.argv[2])
		dist = int(sys.argv[3])
		ilen = int(sys.argv[4])
		c = int(sys.argv[5])
		f = str(sys.argv[6])
		camid = int(sys.argv[7])

		run(path, conf, dist, ilen, c, f, camid)
	else:
		print("Usage:\npython3 pipeline.py <path> <confidence> <radius> <interaction length> <number of processes> <filename> <camid>")
		print("Example:\npython3 pipeline.py 'path/to/data' 0.95 160 3 16 myfilename 0")