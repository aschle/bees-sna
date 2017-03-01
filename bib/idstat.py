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


def working(enu, path, b, e, confidence, year):
    
    df0 = prep.getDF(path, b, e, 0)
    df1 = prep.getDF(path, b, e, 1)
    df2 = prep.getDF(path, b, e, 2)
    df3 = prep.getDF(path, b, e, 3)
    
    data = pd.concat([df0,df1,df2,df3])

    id_list = [0] * 4096
    
    if data.shape[0] == 0:
        print('Empty DF: {} - {}'.format(datetime.datetime.fromtimestamp(b, tz=pytz.UTC), datetime.datetime.fromtimestamp(e, tz=pytz.UTC)))

    else:
        data = prep.calcIds(data, confidence, year)

        df = DataFrame(data.groupby(by="id").size(), columns=["fre"]).reset_index()

        print("{}-{}-{}-{}".format(enu, df.shape, datetime.datetime.fromtimestamp(b, tz=pytz.UTC),datetime.datetime.fromtimestamp(e, tz=pytz.UTC)))
        for r in df.iterrows():
            id_list[r[1].id] = r[1].fre

    return np.array(id_list)


def run(path, ts, confidence=0.95, numCPUs=4, filename="IDS", year=2016, slice_len=5, interval=24):

    pool = multiprocessing.Pool(numCPUs)

    #number of minutes per slice in seconds, for making parallel
    slice_len = slice_len*60   

    begin_ts = ts
    begin_dt = datetime.datetime.fromtimestamp(begin_ts)

    parts = int((interval*60*60)/slice_len)

    print("#Tasks: {}".format(parts))
    
    tasks = []

    for enu, i in enumerate(list(range(parts))):
        b = begin_ts + (i * slice_len)
        e = (b-0.000001) + (slice_len)
        tasks.append((enu, path, b, e, confidence, year))

    results = [pool.apply_async( working, t ) for t in tasks]

    fname = "{}-{}-{}conf-{}h".format(filename, ts, confidence, interval)

    id_list = np.array([0]*4096)

    for result in results:
        res = result.get()
        id_list = id_list+res
        print("Appended Result.")

    Series(id_list).to_csv("{}.csv".format(fname))

if __name__ == '__main__':

    if (len(sys.argv) == 9):
        path = sys.argv[1]

        start = sys.argv[2]
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        start_ts = start_dt.timestamp()

        conf = int(sys.argv[3])/100

        cpus = int(sys.argv[4])

        filename = str(sys.argv[5])

        year = int(sys.argv[6])

        slice_len = int(sys.argv[7])

        interval = int(sys.argv[8])

        print("Path: {}, Start: {}, Confidence: {}, CPUs: {}, Filename: {}, Year: {}, Slice Length: {}, Interval: {}".format(path, start_dt, conf, cpus, filename, year, slice_len, interval))

        run(path, start_ts, conf, cpus, filename, year, slice_len, interval)

    else:
        print("Usage:\npython3 XXX.py <path> <start-date as yyyy-mm-ddThh:mm:ssZ> <confidence in percent> <number of cpus> <filename> <year> <slice_len in minutes> <interval in hours>")
        print("Example:\npython3 XXX.py 'path/to/data' 2016-07-26T00:00:00Z  95 4 myfilname 2016 5 24")