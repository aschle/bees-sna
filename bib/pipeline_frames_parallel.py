
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


if __name__ == '__main__':
    # In[7]:

    p = "/mnt/data/2015082215/"
    c = .99
    dist = 160
    ilen = 6
    cpus = 8

    pool = multiprocessing.Pool(cpus)

    repo = Repository(p)


    number_hours = 1*60*60 #number of hours in total in seconds
    slice_len = 10*60   #number of minutes per slice

    m="08"
    d="22"
    h="15"
    begin = "2015-{}-{}T{}:00:00Z".format(m,d,h) # %Y-%m-%dT%H:%M:%SZ
    begin_ts = datetime.datetime.timestamp(datetime.datetime.strptime(begin, "%Y-%m-%dT%H:%M:%SZ"))

    parts = int(number_hours/slice_len)

    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.1) + (slice_len)
        tasks.append((enu, p, b, e, c, dist, ilen))

    results = [pool.apply_async( generate_network, t ) for t in tasks]


    filename = "testframes-2015082215"


    edges = []

    for result in results:
        edges.append(result.get())
        print("Appended Result.")

    G = prep.create_graph2(pd.concat(edges))
    nx.write_graphml(G, "{}_{}conf_{}dist_{}ilen".format(filename,str(c), str(dist), str(ilen)) + ".graphml")
    print(nx.info(G))