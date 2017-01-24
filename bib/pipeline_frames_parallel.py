import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import preprocessing as prep
import sys  
import datetime
import pytz 

from multiprocessing import Pool
from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple 
from pandas import DataFrame, Series


def generate_network(enu, path, b, e, confidence, distance, ilen):
    
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
        print("#{} DF-{}: {}, {}, {}".format(enu, i, df.shape, datetime.datetime.fromtimestamp(b),datetime.datetime.fromtimestamp(e)))
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


def run(path, confidence=.95, distance=160, interaction_len=3, numCPUs=None, filename="template"):

    p = path
    c = confidence
    dist = distance
    ilen = interaction_len
    cpus = numCPUs

    pool = multiprocessing.Pool(cpus)

    repo = Repository(p)

    slice_len = 5*60   #number of minutes per slice in seconds
    number_hours = 1*60*60 #number of hours in seconds

    it = repo.iter_frames()
    f, fc = it.send(None)
    dt = datetime.datetime.fromtimestamp(f.timestamp, tz=pytz.UTC)

    begin_dt = dt.replace(minute=0, second=0, microsecond=0)
    begin_ts = begin_dt.timestamp()

    end_dt = begin_dt + datetime.timedelta(hours=1)
    end_ts = end_dt.timestamp()

    parts = int(number_hours/slice_len)

    print("#Parts: {}".format(parts))
    
    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.000001) + (slice_len)
        tasks.append((enu, p, b, e, c, dist, ilen))

    results = [pool.apply_async( generate_network, t ) for t in tasks]


    fname = "{}".format(filename)

    edges = []

    for result in results:
        edges.append(result.get())
        print("Appended Result.")

    G = prep.create_graph2(pd.concat(edges))
    nx.write_graphml(G, "{}_{}conf_{}dist_{}ilen".format(fname, str(c), str(dist), str(ilen)) + ".graphml")
    print(nx.info(G))


if __name__ == '__main__':

    if (len(sys.argv) == 7 ):
        path = sys.argv[1]
        conf = float(sys.argv[2])
        dist = int(sys.argv[3])
        ilen = int(sys.argv[4])
        c = int(sys.argv[5])
        f = str(sys.argv[6])

        run(path, conf, dist, ilen, c, f)
    else:
        print("Usage:\npython3 pipeline_frames_parallel.py <path> <confidence> <radius> <interaction length> <number of processes> <filename>")
        print("Example:\npython3 pipeline.py 'path/to/data' 0.95 160 3 16 myfilename")
