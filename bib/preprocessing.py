import pandas as pd
import numpy as np
import networkx as nx

from bb_binary import FrameContainer, Repository, load_frame_container
from pandas import DataFrame, Series


# Eine Datei von einer Kamera holen und ein filecontainer
# zurückgeben
def get_fc(path, camId):
	print("\n### get frame container from {} for cam {}".format(path,camId))
	
	repo = Repository(path)
	file = list(repo.iter_fnames(cam=camId))[0]
	fc = load_frame_container(file)
	return fc


# Create dataframe from framecontainer and return dataframe
def get_dataframe(fc):
	print("\n### get dataframe")

	df = DataFrame()
	
	for f in fc.frames:

		det = DataFrame([d.to_dict() for d in f.detectionsUnion.detectionsDP])
		det['frame_idx'] = f.frameIdx
		det['timestamp'] = f.timestamp
		det['cam_id'] = fc.camId
		det['fc_id'] = fc.id
		det = det.set_index(['fc_id', 'frame_idx', 'idx'])
		df = pd.concat([df, det])

	df.drop(['descriptor',
		'localizerSaliency',
		'xposHive',
		'yposHive',
		'xRotation',
		'radius',
		'yRotation'], axis=1, inplace=True)

	print('Number of Frames: {}'.format(len(df.index.levels[1])))
	print('Number of Detections: {}'.format(df.shape[0]))
	return df


# helper function
def get_binary_bit(bit, threshold):
    if (bit <= threshold):
        return 0
    
    if (bit >= 255-threshold):
        return 1
    
    else:
        return np.nan


# helper function
# Zum ausrechnen der IDs
def get_detected_id(id, threshold):
        
    # Umrechnen in binary array [0,1,1,1,0,1,1,1,0,0,0,1]
    # Ids die nicht umgerechnet werden können, weil außerhalb des threshold, werden NAN
    binary_id = [get_binary_bit(i, threshold) for i in id]

    decimal_id = np.nan

    if not np.isnan(binary_id).any():
        # convert to decimal id using 11 least significant bits
        decimal_id = int(''.join([str(c) for c in binary_id[:11]]), 2)

        # determine what kind of parity bit was used and add 2^11 to decimal id
        # uneven parity bit was used
        if ((sum(binary_id) % 2) == 1):
            decimal_id += 2048

    return decimal_id


# Dezimale ID ausrechnen und an DataFrame angängen
def calcIds(df, threshold):
	print("\n### calc Ids with threshold = {}".format(threshold))
	print('Number of Detections before calcualting IDs: {}'.format(df.shape[0]))
	df['id'] = df.decodedId.apply(get_detected_id, args=(threshold,))
	df = df.drop('decodedId', 1)
	df = df.dropna()
	print('Number of Detections after calcualting IDs: {}'.format(df.shape[0]))
	return df


def get_close_bees(df, distance):
	print("\n### get close ({}) bees".format(distance))
	print('Number of Detections before keeping close bees pairs: {}'.format(df.shape[0]))
	df['key'] = 1
	gr = df.groupby(level = 'frame_idx')

	merged = DataFrame()

	for i, g in gr:
	    cartprodukt = pd.merge(g, g, on='key')
	    merged = pd.concat([merged, cartprodukt])

	merged = merged[merged.id_x < merged.id_y]
	merged = merged.drop('key', axis=1)
	merged.loc[:, 'dist'] = np.sqrt(np.square(merged.xpos_x - merged.xpos_y) \
		+ np.square(merged.ypos_x - merged.ypos_y))

	print('Number of all bee pairs: {}'.format(merged.shape[0]))
	filtered = merged[merged.dist <= distance]
	print('Number of close bee pairs: {}'.format(filtered.shape[0]))
	return filtered


def get_edges(df):
	df = df[['id_x', 'id_y']]
	gr = df.groupby(df.columns.tolist(),as_index=False).size()
	print("Number of unique close bee pairs: {}".format(df.shape[0]))
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


# reshape to columns as timeframes, rows as bee ids
# 		t1  t2  t3 t4 ...
# bee1   1   1   0  1 ...
# bee2   0   0   1  0 ...
# ...
# correct it (fill gaps) and then shape it back again
#
def correct_bees_timeframes(df):

	gr = df.groupby(level='frame_idx')

	# get all unique ids von allen Frames
	u_id = df.id.unique()

	# das mit der 1024 ändern, das ist so doof
	dft = DataFrame(0, index=u_id, columns=np.arange(1024))

	for i, group in gr:
		l = group['id']
		dft.loc[l,i] = 1

	print("Reshaped to columns as timeframes and rows as bees!")
	print(dft.shape)

	# Correct gaps, fill gaps of length 1 with a one
	# 101001101 -> 111001111

	dft = dft.apply(fill_gaps, axis=1)
	#dft['total'] = dft.sum(axis=1)

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



	return dft

def create_graph(gr, filename):
	G = nx.Graph()
	df = DataFrame(gr)
	df = df.reset_index(level=['id_x', 'id_y'])

	for row in df.itertuples():
		G.add_edge(int(row[0]), int(row[1]), weight=int(row[2]))
	
	print(nx.info(G))

	nx.write_graphml(G, filename + ".graphml")

	return G

###########
###########

# path = "../00_Data/testset_2015_1h/2015102215"
path = "../00_Data/testset_2015_1h/2015082215"
# path = "../00_Data/testset_2015_1h/2015092215"

fc = get_fc(path, 0)
df = get_dataframe(fc)
df = calcIds(df, 30)

#df = correct_bees_timeframes(df)

df = get_close_bees(df, 150)
df = get_edges(df)	

G = create_graph(df, "2015082215")

print(df.head(10))
