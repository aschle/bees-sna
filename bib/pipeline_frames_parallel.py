
# coding: utf-8

# In[6]:

import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import preprocessing as prep
import sys
import datetime

from multiprocessing import Pool
from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple 
from pandas import DataFrame, Series


def generate_network(enu, path, b, e, confidence, distance, ilen):
    print("path: {}".format(enu), path,b,e,confidence,distance,ilen)
    
    repo = Repository(path)
    xmax = 3000
    offset = 200
    
    Detection = namedtuple('Detection',
                           ['idx', 'xpos', 'ypos', 'radius', 'zRotation', 'decodedId', 'frame_idx', 'timestamp', 'cam_id', 'fc_id'])

    # one df per cam
    parts = np.empty(4, dtype=object)

    for i in list(range(4)):
         
        tpls = []
        myid = 0
        for frame, fc in repo.iter_frames(begin=b, end=e, cam=i):
            for d in frame.detectionsUnion.detectionsDP:
                d = Detection(d.idx, d.xpos, d.ypos, d.radius, d.zRotation, list(d.decodedId), myid, frame.timestamp, fc.camId, fc.id)
                tpls.append(d)
            myid += 1
        
        df = DataFrame(tpls)
        df = prep.calcIds(df,confidence)
        parts[i] = df
        

    # cam 0 und cam1 nach rechts verschieben
    parts[0].xpos = parts[0].xpos + xmax + offset
    parts[1].xpos = parts[1].xpos + xmax + offset

    # Seiten zusammenfugen
    side0 = pd.concat([parts[3], parts[0]])
    side1 = pd.concat([parts[2], parts[1]])

    close1 = prep.get_close_bees_ckd(side0, distance)
    close2 = prep.get_close_bees_ckd(side1, distance)

    close = pd.concat([close1,close2])

    p = prep.bee_pairs_to_timeseries(close)

    return prep.extract_interactions(p,ilen)


def run(path, month, day, hour, confidence=.95, distance=160, interaction_len=3, c=None, filename="template"):

    p = path
    c = confidence
    dist = distance
    ilen = interaction_len
    cpus = c

    pool = multiprocessing.Pool(cpus)

    repo = Repository(p)


    number_hours = 1*60*60 #number of hours in total in seconds
    slice_len = 6*60   #number of minutes per slice

    m = month
    d = day
    h = hour
    begin = "2015-{}-{}T{}:00:00Z".format(m,d,h) # %Y-%m-%dT%H:%M:%SZ
    begin_ts = datetime.datetime.timestamp(datetime.datetime.strptime(begin, "%Y-%m-%dT%H:%M:%SZ"))

    parts = int(number_hours/slice_len)

    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.1) + (slice_len)
        tasks.append((enu, p, b, e, c, dist, ilen))

    results = [pool.apply_async( generate_network, t ) for t in tasks[:1]]


    filename = "{}-{}-{}-1h-allCams"


    edges = []

    for result in results:
        edges.append(result.get())
        print("Appended Result.")

    G = prep.create_graph2(pd.concat(edges))
    nx.write_graphml(G, "{}_{}conf_{}dist_{}ilen".format(filename, str(c), str(dist), str(ilen)) + ".graphml")
    print(nx.info(G))


if __name__ == '__main__':

    if (len(sys.argv) == 10 ):
        path = sys.argv[1]
        m = sys.argv[2]
        d = sys.argv[3]
        h = sys.argv[4]
        conf = float(sys.argv[5])
        dist = int(sys.argv[6])
        ilen = int(sys.argv[7])
        c = int(sys.argv[8])
        f = str(sys.argv[9])

        run(path, m, d, h, conf, dist, ilen, c, f)
    else:
        print("Usage:\npython3 pipeline_frames_parakkek.py <path> <month> <day> <hour> <confidence> <radius> <interaction length> <number of processes> <filename>")
        print("Example:\npython3 pipeline.py 'path/to/data' '08' '21' '15' 0.95 160 3 16 myfilename")