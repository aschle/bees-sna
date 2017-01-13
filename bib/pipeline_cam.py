import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import preprocessing as prep
import sys
import itertools

from multiprocessing import Pool
from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple 
from pandas import DataFrame, Series

def get_files(path, camid):
	repo = Repository(path)
	files = list(repo.iter_fnames(cam=camid))
	return files

def generate_network(index, file, confidence, ilens, distances):

    
	fc = load_frame_container(file)
	df = prep.get_dataframe2(fc)
	df = prep.calcIds(df,confidence)

	result = []

	for distance in distances:
		close = prep.get_close_bees_ckd(df, distance)
		p = prep.bee_pairs_to_timeseries(close)

		for ilen in ilens:
			r = prep.extract_interactions(p,ilen)
			restult.append((ilen,distance,r))

	return result


def run(p, conf=.95, c=None, filename="template", camid=0):
	path = p
	confidence = conf
	cpus = c

	# make file file packges
	files = get_files(path, camid)

	# how many cpus
	if (cpus == None):
		max_num_cpu = multiprocessing.cpu_count()
		cpus = max_num_cpu - (max_num_cpu/2/2)

	pool = multiprocessing.Pool(cpus)

	ilens = [3,6,9,12,15,18,21] #ilen
	distances = [100,110,120,130,140,150,160] #distance
	l = list(itertools.product(ilens,distances))

	tasks = []

	for e, f in enumerate(files):
		tasks.append((e, f, confidence, ilens, distances))

	results = [pool.apply_async( generate_network, t ) for t in tasks]

	# Egebnisse zusammenbasteln
	edges = {item:[] for item in l}

	for result in results:
		# ein result ist eine liste von ganz vielen (d,i,r)s
		reslist = result.get()

		for i,d,r in reslist:
			edges[(i,d)].extend(r)

	for i,d in edges:
		G = prep.create_graph2(pd.concat(edges[(i,d)]))
		nx.write_graphml(G, "{}_{}conf_{}dist_{}ilen_cam{}.graphml".format(
			filename,
			str(confidence),
			str(d),
			str(i),
			str(camid)))

		print(nx.info(G))


if __name__ == '__main__':

	if (len(sys.argv) == 6 ):
		path = sys.argv[1]		
		conf = float(sys.argv[2])
		c = int(sys.argv[3])
		f = str(sys.argv[4])
		camid = int(sys.argv[5])

		run(path, conf, c, f, camid)
	else:
		print("Usage:\npython3 pipeline.py <path> <confidence> <number of processes> <filename> <camid>")
		print("Example:\npython3 pipeline.py 'path/to/data' 0.95 16 myfilename 0")