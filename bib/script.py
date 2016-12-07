def f(c,d,m,l):
	#!/usr/bin/python

	import sys
	import preprocessing as prep
	import pandas as pd
	import numpy as np
	import networkx as nx

	# python script.py <confidence> <distance> <month> <length>
	# python3 script.py 0.9 160 08 3

	# c = float(sys.argv[1])
	# d = int(sys.argv[2])
	# m = sys.argv[3]
	# l = int(sys.argv[4])

	filename = "{}month-{}dist-{}conf-{}len".format(m,d,c,l)

	f = "../00_Data/testset_2015_1h/"
	p = "2015" + m + "2215"

	CONFIDENCE = c
	DISTANCE = d

	xmax = 3000
	ymax = 4000
	LENGTH = l

	path = f+p

	fc0 = prep.get_fc(path,0)
	fc1 = prep.get_fc(path,1)
	fc2 = prep.get_fc(path,2)
	fc3 = prep.get_fc(path,3)


	df3 = prep.get_dataframe(fc3)
	df3 = prep.calcIds(df3,CONFIDENCE)
	df0 = prep.get_dataframe(fc0)
	df0 = prep.calcIds(df0,CONFIDENCE)

	df2 = prep.get_dataframe(fc2)
	df2 = prep.calcIds(df2,CONFIDENCE)
	df1 = prep.get_dataframe(fc1)
	df1 = prep.calcIds(df1,CONFIDENCE)

	df0.xpos = df0.xpos + xmax
	df1.xpos = df1.xpos + xmax

	side0 = pd.concat([df3, df0])
	side1 = pd.concat([df2, df1])

	close1 = prep.get_close_bees(side0, DISTANCE)
	close2 = prep.get_close_bees(side1, DISTANCE)

	close = pd.concat([close1,close2])

	p = prep.bee_pairs_to_timeseries(close)

	i = prep.extract_interactions(p,LENGTH)

	G = prep.create_graph2(i)

	nx.write_graphml(G, filename + ".graphml")