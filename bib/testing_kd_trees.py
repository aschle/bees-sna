import pandas as pd
import numpy as np
import networkx as nx
import preprocessing as prep

from bb_binary import FrameContainer, Repository, load_frame_container
from pandas import DataFrame, Series


path1 = "../00_Data/testset_2015_1h/2015082215"
path2 = "../00_Data/testset_2015_1h/2015092215"
path3 = "../00_Data/testset_2015_1h/2015102215"

path = path3

fc = get_fc(path, 0)
df = get_dataframe(fc)
df = calcIds(df, CONFIDENCE)



