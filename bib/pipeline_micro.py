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

# packt immer die 4 Kameradateien zusammen
def get_files(path):
    repo = Repository(path)
    file = list(repo.iter_fnames())
    a = [f.split('/')[-1].split("_")[1] for f in file]
    l = len(a)/4
    npa = np.array(file).reshape(int(l),4)
    return npa

def generate_networks(index, file_list, confidence=.95, distance=160, ilen=3, window_size=256):
    print("process {} - start".format(index))
    
    xmax = 3000
    
    # list of networks
    network_list = []
    
    # one df per cam
    dataframes = np.empty(4, dtype=object)

    for i in list(range(4)):
        fc = load_frame_container(file_list[i])
        df = prep.get_dataframe2(fc)
        df = prep.calcIds(df,confidence)

        camIdx = int(file_list[i].split("/")[-1].split("_")[1])
        dataframes[camIdx] = df

    # cam 0 und cam1 nach rechts verschieben
    dataframes[0].xpos = dataframes[0].xpos + xmax
    dataframes[1].xpos = dataframes[1].xpos + xmax

    # Seiten zusammenfugen
    side0 = pd.concat([dataframes[3], dataframes[0]])
    side1 = pd.concat([dataframes[2], dataframes[1]])

    close1 = prep.get_close_bees_ckd(side0, distance)
    close2 = prep.get_close_bees_ckd(side1, distance)

    close = pd.concat([close1,close2])

    p = prep.bee_pairs_to_timeseries(close)
    
    for w in list(range(int(1024/window_size))): 
        part = p.ix[:,window_size*w:window_size*(w+1)]
        edges = prep.extract_interactions(part,ilen)
        g = prep.create_graph2(edges)
        network_list.append(((index*1024)+(w*window_size),g))
    
    print("process {} - end - {}".format(index, len(network_list)))
    return network_list

def run(p, conf=.95, dist=160, ilen=3, c=None, filename="template", ws=256):
	path = p
	confidence = conf
	distance = dist
	interaction_len = ilen
	cpus = c
	window_size = ws

	# make file file packges
	files = get_files(path)


	# how many cpus
	if (cpus == None):
		max_num_cpu = multiprocessing.cpu_count()
		cpus = max_num_cpu - (max_num_cpu/2/2)

	pool = multiprocessing.Pool(cpus)

	tasks = []

	for e, f in enumerate(files):
		tasks.append((e, f, confidence, distance, interaction_len, window_size))

	results = [pool.apply_async( generate_networks, t ) for t in tasks]

	# Egebnisse zusammenbasteln
	networks = []

	for result in results:
		networks.extend(result.get())

	for i,g in networks:
		nx.write_graphml(g, "micro-networks/{}_{}ws_{}conf_{}dist_{}ilen_{}".format(filename, window_size, str(confidence), str(distance), str(interaction_len), i) + ".graphml")

	print("Ende: {}".format(len(networks)))

if __name__ == '__main__':

	if (len(sys.argv) == 8 ):
		path = sys.argv[1]		
		conf = float(sys.argv[2])
		dist = int(sys.argv[3])
		ilen = int(sys.argv[4])
		c = int(sys.argv[5])
		f = str(sys.argv[6])
		ws = int(sys.argv[7])

		run(path, conf, dist, ilen, c, f, ws)
	else:
		print("Usage:\npython3 pipeline.py <path> <confidence> <radius> <interaction length> <number of processes> <filename-prefix> <window-size>")
		print("Example:\npython3 pipeline.py 'path/to/data' 0.95 160 3 16 myfilename-prefix 256")