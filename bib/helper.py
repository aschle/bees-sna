import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import sys

from multiprocessing import Pool
from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple 
from pandas import DataFrame, Series


# packt immer die 4 Kameradateien zusammen
def get_files(path):
	repo = Repository(path)
	file = list(repo.iter_fnames())
	a = [f.split('/')[-1].split("_")[1] for f in file]
	l = len(a)/4
	npa = np.array(file).reshape(int(l),4)
	return npa