import pandas as pd
import numpy as np
import networkx as nx

from bb_binary import FrameContainer, Repository, load_frame_container
from pandas import DataFrame, Series
from scipy import spatial
from collections import namedtuple


# Eine Datei von einer Kamera holen und ein filecontainer
# zurueckgeben
def get_fc(path, camId):
	print("file {} cam {}".format(path,camId))
	
	repo = Repository(path)
	file = list(repo.iter_fnames(cam=camId))[0]
	fc = load_frame_container(file)
	return fc


# Create dataframe from framecontainer and return dataframe
def get_dataframe(fc):
	
	detection = namedtuple('Detection', ['idx','xpos','ypos',
		'radius','decodedId', 'frame_idx', 'timestamp', 'cam_id', 'fc_id'])

	l = []
	for f in fc.frames:
		tpls = [detection(d.idx, d.xpos, d.ypos, d.radius, list(d.decodedId),
			f.frameIdx, f.timestamp, fc.camId, fc.id)
			for d in f.detectionsUnion.detectionsDP]
		l.append(pd.DataFrame(tpls))
	return pd.concat(l)


# helper function
# Zum ausrechnen der IDs
def get_detected_id(bits):

	# Umrechnen in binary array [0,1,1,1,0,1,1,1,0,0,0,1]
	bits = np.array(bits)
	bits = bits/255
	binary_id = [int(x > 0.5) for x in bits]

	decimal_id = int(''.join([str(c) for c in binary_id[:11]]), 2)

	# determine what kind of parity bit was used and add 2^11 to decimal id
	# uneven parity bit was used
	if ((sum(binary_id) % 2) == 1):
		decimal_id += 2048

	return decimal_id


def get_confidence(bits):
	# 12 bits mit Werten zwischen 0 und 256
	bits = np.array(bits)
	bits = bits/255

	return np.min(np.abs(0.5 - bits)) * 2

# Dezimale ID ausrechnen und an DataFrame angaengen
def calcIds(df, threshold):
	# print('\n### Calc IDs with threshold: {}'.format(threshold))
	#print('#Detections before calcualting IDs: {}'.format(df.shape[0]))

	# calc confidence value
		# 0...256 in 0...1 umrechnen
		# fuer jedes bit abstand zu 0.5 berechnen und dann minimum behalten
	# add confidence value to dataframe as column
	df = df.assign(confidence = df.decodedId.apply(get_confidence))

	# die detections entfernen die nicht  die nicht gut genug sind
	df = df[df.confidence >= threshold]

	# fuer den Rest der ueber bleibt die ID berechnen und an DF anhaengen
	df = df.assign(id = df.decodedId.apply(get_detected_id))

	df = df.drop('decodedId', 1)

	#print('Number of Detections after calcualting IDs: {}'.format(df.shape[0]))
	return df

def get_close_bees(df, distance):

	df = df.reset_index(level = 'frame_idx')

	m = pd.merge(df, df, on='frame_idx')
	#m = m.query('id_x < id_y')
	m = m[m.id_x < m.id_y]

	m.loc[:, 'dist'] = np.square(m.xpos_x - m.xpos_y) \
		+ np.square(m.ypos_x - m.ypos_y)

	filtered = m[m.dist <= distance**2]

	filtered = filtered[['frame_idx','id_x', 'id_y']]
	return filtered

# Depricated
def get_close_bees_kd(df, distance):

	df_close = DataFrame()

	gr = df.groupby('frame_idx')

	for i, group in gr:
		xy_coordinates = group[['xpos', 'ypos']].values
		tree = spatial.KDTree(xy_coordinates, leafsize=20)
		result = tree.query_pairs(distance)
		l = [[i,group['id'].iat[a], group['id'].iat[b]] for a,b in result]
		df_close = df_close.append(DataFrame(l, columns=['frame_idx', 'id_x', 'id_y']))

	return df_close

def get_ketten(kette, val):
    kette = kette.apply(str)
    s = kette.str.cat(sep='')
    ss = s.split('0')
    return [x for x in ss if len(x) > 0]

def bee_pairs_to_timeseries(df):
	close = df[['frame_idx', 'id_x', 'id_y']]
	close = close.set_index(['frame_idx'])
	close['pair'] = list(zip(close.id_x, close.id_y))
	u_pairs = close.pair.unique()
	dft = DataFrame(0, index=u_pairs, columns=np.arange(1024))
	gr = close.groupby(level='frame_idx')

	for i, group in gr:
		l = group['pair']
		dft.loc[l,i] = 1

	return dft

def extract_interactions(dft, minlength):
    kette = dft.apply(get_ketten, axis=1, args=[1])
    kk = kette.apply(lambda x: [len(item) for item in x])
    kk = kk.apply(lambda x: len([item for item in x if item >= minlength]))
    return kk[kk > 0]

def get_edges(df):
	df = df[['id_x', 'id_y']]
	gr = df.groupby(df.columns.tolist(), as_index=False).size()
	print("Number of unique close bee pairs: {}".format(gr.shape[0]))
	return gr

# fills gaps of:
# 101010011 ->
# 111110011
def fill_gaps(ll):
    for n,k in enumerate(ll[:-2]):
        left = k
        right = ll[n+2]
        m = ll[n+1]

        if (left + right == 2):
            ll[n+1] = 1
    return ll


def df_to_timeseries(df):
	gr = df.groupby(level='frame_idx')
	num_columns = len((df.index.get_level_values('frame_idx')).unique())

	# levels wegschmeissen
	df = df.reset_index(level = ['fc_id', 'frame_idx', 'idx'], drop=False )

	# get all unique ids von allen Frames
	u_id = df.id.unique()

	dft = DataFrame(0, index=u_id, columns=np.arange(num_columns))

	for i, group in gr:
		l = group['id']
		dft.loc[l,i] = 1

	return dft


def timeseries_to_df(dft, df):

	# Zurueckumwandeln in urspruegliches Format: Dabei muessen aber die neu 
	# entstandenen Bienen zu bestimmten Zeitpunkten eingebaut werden.
	#
	# t1  bee1 xpos ypos ...
	#	  bee2 xpos ypos ...
	#     ...  ...  ...
	#		
	# t2  bee2 xpos ypos ...
	#     bee3 xpos ypos ...
	#     ...

	df = df.reset_index(['frame_idx', 'fc_id', 'idx'])
	final = DataFrame()

	for col in list(range(dft.shape[1])):
	
		# die indexes wo ne eins steht merken
		l = dft[dft[col] == 1].index.tolist()

		for item in l:
			# print("{}-{}".format(col,item))
			# element zum timeframe rausholen
			tfe = df[df.frame_idx == col]
			if (tfe[tfe['id'] == item].shape[0] > 0):            
				final = pd.concat([final, tfe[tfe['id'] == item]])
			else:
				pre = df[df.frame_idx == col-1]
				predict = pre[pre['id'] == item]
	            
				post = df[df.frame_idx == col+1]
				postdict = post[post['id'] == item]
	            
				x = (list(predict['xpos'])[0] + list(postdict['xpos'])[0])/2
				y = (list(predict['ypos'])[0] + list(postdict['ypos'])[0])/2
				row = pd.DataFrame({
					'frame_idx': col,
					'id':item,
					'xpos':x,
					'ypos':y,
					'timestamp': list(predict['timestamp'])[0],
					'fc_id': list(predict['fc_id'])[0],
					'cam_id': list(predict['cam_id'])[0]},
					index = [col])
				final = final.append(row)

	final = final.set_index(['frame_idx'])

	return final


# reshape to columns as timeframes, rows as bee ids
# 		t1  t2  t3 t4 ...
# bee1   1   1   0  1 ...
# bee2   0   0   1  0 ...
# ...
# correct it (fill gaps) and then shape it back again
#
def correct_bees_timeframes(df):
	dft = df_to_timeseries(df)
	dft = dft.apply(fill_gaps, axis=1)
	df_corrected = timeseries_to_df(dft, df)
	return df_corrected


def create_graph(gr, filename):
	G = nx.Graph()
	df = DataFrame(gr)
	df = df.reset_index(level=['id_x', 'id_y'])

	for row in df.itertuples():
		print(row)
		G.add_edge(int(row[1]), int(row[2]), weight=int(row[3]))
	
	print(nx.info(G))

	nx.write_graphml(G, filename + ".graphml")

	return G

def create_graph2(pairs):
	G = nx.Graph()

	for elem in pairs.iteritems():
		G.add_edge(int(elem[0][0]), int(elem[0][1]), weight=int(elem[1]))
	#print(nx.info(G))

	# nx.write_graphml(G, filename + ".graphml")
	return G


###########
###########

if __name__ == '__main__':

	CONFIDENCE = 0.9
	DISTANCE = 160
	CAMS = 4

	path1 = "../00_Data/testset_2015_1h/2015082215"
	path2 = "../00_Data/testset_2015_1h/2015092215"
	path3 = "../00_Data/testset_2015_1h/2015102215"

	l = [path1, path2, path3]
	l = [path3]

	for path in l:

		pairs = Series()

		for i in list(range(CAMS)):
			fc = get_fc(path, i)
			df = get_dataframe(fc)
			df = calcIds(df, CONFIDENCE)

			#df = correct_bees_timeframes(df)

			df = get_close_bees_old(df, DISTANCE)
			print(df.shape)

			p = bee_pairs_to_timeseries(df)
			print(p.shape[0])
			pairs = pairs.append(p)

			## macht die haufigkeit *wie viel frames als gewicht
			##df = get_edges(df)	

			#G = create_graph(df, "2015082215")
		
		G = create_graph2(pairs, "2015102215-6min-{}cams-{}-{}-3".format(CAMS, DISTANCE, str(CONFIDENCE).replace('.','')))

		#print(pairs)
