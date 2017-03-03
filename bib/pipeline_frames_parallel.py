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


def generate_network(enu, path, b, e, confidence, distance, ilen, year, gap):
    
    xmax = 3000
    offset = 2 * distance
    
    parts = np.empty(4, dtype=object)

    abbrechen = False

    stat = []

    # one df per camera
    for i in list(range(4)):

        df = prep.getDF(path, b, e, i)

        numframes = 0
        if (df.shape[0] != 0):
            numframes = df.groupby(by='frame_idx').size().shape[0]

        stat.append(numframes)

        df = prep.calcIds(df, confidence, year)

        # Abbrechen, wenn ein DF leer ist
        if(df.shape[0] == 0):
            abbrechen = True

        parts[i] = df

    if abbrechen == True:
        print("#{}: From {} to {} - {}".format(enu, datetime.datetime.fromtimestamp(b, tz=pytz.UTC), datetime.datetime.fromtimestamp(e, tz=pytz.UTC), stat))
        return Series()
    
    if year == 2015:
        # cam 0 und cam1 nach rechts verschieben
        parts[0].xpos = parts[0].xpos + xmax + offset
        parts[1].xpos = parts[1].xpos + xmax + offset

        # Seiten zusammenfugen
        side0 = pd.concat([parts[3], parts[0]])
        side1 = pd.concat([parts[2], parts[1]])

    if year == 2016:
        # cam 1 und cam 3 nach rechts verschieben
        parts[1].xpos = parts[1].xpos + xmax + offset
        parts[3].xpos = parts[3].xpos + xmax + offset

        # Syncronisieren der Kameras pro Seite
        parts[0], parts[1]= prep.mapping(parts[0], parts[1])
        parts[2], parts[3]= prep.mapping(parts[2], parts[3])

        d0  = len(parts[0].frame_idx.unique())
        d1  = len(parts[1].frame_idx.unique())
        d2  = len(parts[2].frame_idx.unique())
        d3  = len(parts[3].frame_idx.unique())

        print("#{}: From {} to {} - {} - {} {} {} {}".format(enu, datetime.datetime.fromtimestamp(b, tz=pytz.UTC), datetime.datetime.fromtimestamp(e, tz=pytz.UTC), stat, d0,d1,d2,d3))

        # Seiten zusammenfugen
        side0 = pd.concat([parts[0], parts[1]])
        side1 = pd.concat([parts[2], parts[3]])

    
    dt = datetime.datetime.fromtimestamp(b, tz=pytz.UTC)
    # Detectionen wegschmeißen, dessen ID insgesamt sehr wenig detektiert wurde
    side0 = prep.removeDetectionsList(side0, dt.strftime("%Y-%m-%d"))
    side1 = prep.removeDetectionsList(side1, dt.strftime("%Y-%m-%d"))

    close1 = prep.get_close_bees_ckd(side0, distance)
    close2 = prep.get_close_bees_ckd(side1, distance)

    close = pd.concat([close1,close2])


    # Zeitreihe für Paare machen
    p = prep.bee_pairs_to_timeseries(close)

    # Coorect pair time series
    p_corrected = p.apply(prep.fill_gaps, axis=1, args=[gap])

    return prep.extract_interactions(p_corrected,ilen)


def run(path, start_ts, network_size, confidence=.95, distance=160, interaction_len=3, numCPUs=None, filename="template", year=2016, gap=2):

    pool = multiprocessing.Pool(numCPUs)

    #number of minutes per slice in seconds, for making parallel
    slice_len = 5*60   # TODO make parameter

    #network_size in seconds
    size = network_size*60 

    begin_ts = start_ts
    begin_dt = datetime.datetime.fromtimestamp(begin_ts)

    parts = int(size/slice_len)

    print("#Parts: {}".format(parts))
    
    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.000001) + (slice_len)
        tasks.append((enu, path, b, e, confidence, distance, interaction_len, year, gap))

    results = [pool.apply_async( generate_network, t ) for t in tasks]


    fname = "{}".format(filename)

    edges = []

    for result in results:
        res = result.get()

        if res.empty:
            print("Not Appended.")
        else:
            edges.append(res)
            print("Appended Result.")
            

    G = prep.create_graph2(pd.concat(edges))

    nx.write_graphml(
        G,
        "{}_{}conf_{}dist_{}ilen_{}gap_{}minutes_{}.graphml".format(fname, str(int(confidence*100)), str(distance), str(interaction_len), str(gap), str(network_size), str(datetime.datetime.fromtimestamp(start_ts, tz=pytz.UTC))))

    print(nx.info(G))


if __name__ == '__main__':

    if (len(sys.argv) == 11):
        path = sys.argv[1]

        start = sys.argv[2]
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        start_ts = start_dt.timestamp()

        size = int(sys.argv[3])

        conf = int(sys.argv[4])/100
        dist = int(sys.argv[5])
        ilen = int(sys.argv[6])
        c = int(sys.argv[7])
        f = str(sys.argv[8])
        year = int(sys.argv[9])
        gap = int(sys.argv[10])

        print("ilen {}, conf {}, dist {}, gap {}, minutes {}, start {}". format(ilen, int(conf*100), dist, gap, size, start))
        run(path, start_ts, size, conf, dist, ilen, c, f, year, gap)

    else:
        print("Usage:\npython3 pipeline_frames_parallel.py <path> \
            \n<start-date as yyyy-mm-ddThh:mm:ssZ> <network_size in minutes> \
            \n<confidence> <radius> <interaction length> <number of processes> \
            \n<filename> <year> <gap-size>")
        print("Example:\npython3 pipeline.py 'path/to/data' 2016-07-26T00:00:00Z 60 \
            \n95 212 3 8 myfilename 2016 2")