from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple
from collections import Counter
from pandas import DataFrame, Series

import multiprocessing
import sqlite3
import numpy as np
import pandas as pd
import sys

def get_detected_id(bits):
    binary_id = (bits>0.5)*1
    decimal_id = int(''.join([str(c) for c in binary_id.tolist()[:11]]), 2)
    if ((sum(binary_id) % 2) == 1):
        decimal_id += 2048

    return decimal_id

def get_confidence(bits):
    return np.min(np.abs(0.5 - bits)) * 2

# Dezimale ID ausrechnen und an DataFrame angaengen
def calcIds(df, threshold):
    df.decodedId = df.decodedId.apply(lambda x: np.array(x)/255)
    df = df.assign(confidence = df.decodedId.apply(get_confidence))
    df = df[df.confidence >= threshold]
    df = df.assign(id = df.decodedId.apply(get_detected_id))

    return df

Detection = namedtuple(
    'Detection',
    ['idx', 'xpos', 'ypos', 'radius', 'zRotation', 'decodedId', 'frame_idx', 'timestamp', 'cam_id', 'fc_id']
)

def get_dataframe2(fcs):
    """
    Converts framecontainer(s) to a dataframe. Uses fixed columns.
    :param fcs: One or multiple framecontainer
    :return: Pandas Dataframe
    """
    if not isinstance(fcs, (tuple, list)):
        fcs = fcs,

    tpls = []
    for fc in fcs:
        for f in fc.frames:
            for d in f.detectionsUnion.detectionsDP:
                d = Detection(d.idx, d.xpos, d.ypos, d.radius, d.zRotation, list(d.decodedId), f.frameIdx, f.timestamp, fc.camId, fc.id)
                tpls.append(d)

    df = pd.DataFrame(tpls)
    return df

def getIDs(m, d, h, files, conf):
    a = np.zeros(2**12, dtype='int32')
    
    for f in files:
        fc = load_frame_container(f)
        df = get_dataframe2(fc)
        df = calcIds(df, conf)
        ids = list(df.id)
        for i in ids:
            a[i] += 1
    return (m,d,h,a)

def run(path, conf, cpus, year):
    
    pool = multiprocessing.Pool(cpus)
    
    repo = Repository(path)
    
    file_list = []
    for f in repo.iter_fnames():
        string = f.split(year)
        datum = string[2].split("/")
        file_list.append([datum[1], datum[2], datum[3], f])
    
    # DataFrame with all files
    df = DataFrame(file_list, columns=['m', 'd', 'h', 'file'])
    
    # Group by hours
    gr = df.groupby(by=['m','d','h'])
    
    # Pro Gruppe einen Task erstellen
    tasks = []

    for group in gr:
        files = list(group[1].file)
        tasks.append((group[0][0], group[0][1], group[0][2], files, conf))
        
    results = [pool.apply_async( getIDs, t ) for t in tasks]
    
    # Write results straight to DB
    conn = sqlite3.connect('ids.db')
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS IDS''')
    c.execute('''CREATE TABLE IDS
           (MONTH   CHARACTER(20)   NOT NULL,
           DAY   CHARACTER(20)   NOT NULL,
           HOUR  CHARACTER(20)   NOT NULL,
           ID   INT   NOT NULL,
           COUNT   INT   NOT NULL);''')
    
    for result in results:
        
        res = result.get()
        m = res[0]
        d = res[1]
        h = res[2]
        
        for e, r in enumerate(res[3]):
            c.execute("insert into ids (month, day, hour, id, count) values (?, ?, ?, ?, ?)",
            (m, d, h, e, int(r)))

    pool.close()
    pool.join()
    
    counting = c.execute('SELECT count(*) FROM IDS')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':

	if (len(sys.argv) == 5 ):
		path = sys.argv[1]		
		conf = float(sys.argv[2])
		cpus = int(sys.argv[3])
		year = sys.argv[4]

		run(path, conf, cpus, year)
	else:
		print("Usage:\npython3 ids_script.py <path> <confidence> <number of processes>")
		print("Example:\npython3 ids_script.py 'path/to/data' 0.95 8")
