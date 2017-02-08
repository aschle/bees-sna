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


def generate_network(enu, path, b, e, confidence, distance, ilen, year):
    
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
        print("#{} DF-{}: {}, {}, {}".format(enu, i, df.shape, datetime.datetime.fromtimestamp(b, tz=pytz.UTC),datetime.datetime.fromtimestamp(e, tz=pytz.UTC)))
        df = prep.calcIds(df,confidence, year)
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


def run(path, start_ts, network_size, confidence=.95, distance=160, interaction_len=3, numCPUs=None, filename="template", year=2015):

    p = path
    c = confidence
    dist = distance
    ilen = interaction_len
    cpus = numCPUs
    y = year

    pool = multiprocessing.Pool(cpus)

    repo = Repository(p)

    #number of minutes per slice in seconds, for making parallel
    slice_len = 5*60   

    #network_size in seconds
    size = network_size*60 

    it = repo.iter_frames()
    f, fc = it.send(None)
    dt = datetime.datetime.fromtimestamp(f.timestamp, tz=pytz.UTC)

    begin_ts = start_ts
    begin_dt = datetime.datetime.fromtimestamp(begin_ts)

    parts = int(size/slice_len)

    print("#Parts: {}".format(parts))
    
    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.000001) + (slice_len)
        tasks.append((enu, p, b, e, c, dist, ilen, y))

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

    if (len(sys.argv) == 10):
        path = sys.argv[1]

        start = sys.argv[2]
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        start_ts = start_dt.timestamp()

        size = int(sys.argv[3])

        conf = float(sys.argv[4])
        dist = int(sys.argv[5])
        ilen = int(sys.argv[6])
        c = int(sys.argv[7])
        f = str(sys.argv[8])
        year = int(sys.argv[9])

        run(path, start_ts, size, conf, dist, ilen, c, f, year)

    else:
        print("Usage:\npython3 pipeline_frames_parallel.py <path> <start-date as yyyy-mm-ddThh:mm:ssZ> <network_size in minutes> <confidence> <radius> <interaction length> <number of processes> <filename> <year>")
        print("Example:\npython3 pipeline.py 'path/to/data' 2015-08-21T00:00:00Z 60 0.95 160 3 16 myfilename 2015")
