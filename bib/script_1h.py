import sys
import preprocessing as prep
import pandas as pd
import numpy as np
import networkx as nx


c = float(sys.argv[1])
d = int(sys.argv[2])
m = sys.argv[3]
l = int(sys.argv[4])

filename = "{}month-{}dist-{}conf-{}len-1h".format(m,d,str(c).replace('.',''),l)

f = "../../data/1h/"
p = "2015" + m + "2215"

CONFIDENCE = c
DISTANCE = d

xmax = 3000
ymax = 4000
LENGTH = l

path = f+p


def get_files(path):
    repo = Repository(path)
    file = list(repo.iter_fnames())
    a = [f.split('/')[-1].split("_")[1] for f in file]
    l = len(a)/4
    npa = np.array(file).reshape(l,4)
    return npa



files = get_files(path)

interactions = Series()


for file_list in files[:2]:
    
    print(file_list)
    
    dataframes = np.empty(4, dtype=object)
    
    for i in list(range(4)):
        print(i)
        fc = load_frame_container(file_list[i])
        df = prep.get_dataframe(fc)
        df = prep.calcIds(df,CONFIDENCE)

        camIdx = file_list[i].split("/")[-1].split("_")[1]
        dataframes[camIdx] = df
    
    # cam 0 und cam1 nach rechts verschieben
    dataframes[0].xpos = dataframes[0].xpos + xmax
    dataframes[1].xpos = dataframes[1].xpos + xmax

    # Seiten zusammenfugen
    side0 = pd.concat([dataframes[3], dataframes[0]])
    side1 = pd.concat([dataframes[2], dataframes[1]])

    close1 = prep.get_close_bees(side0, DISTANCE)
    close2 = prep.get_close_bees(side1, DISTANCE)

    close = pd.concat([close1,close2])
    
    p = prep.bee_pairs_to_timeseries(close)

    edges = prep.extract_interactions(p,LENGTH)
    
    interactions = pd.concat([interactions, edges])

G = prep.create_graph2(interactions)
print(nx.info(G))

nx.write_graphml(G, filename + ".graphml")