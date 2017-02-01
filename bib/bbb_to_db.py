from bb_binary import FrameContainer, Repository, load_frame_container
from collections import namedtuple
from collections import Counter
from pandas import DataFrame, Series
import preprocessing as prep

import multiprocessing
import sqlite3
import numpy as np
import pandas as pd
import sys
import datetime
import pytz

Detection = namedtuple(
	'Detection',
	['frame_id', 'xpos', 'ypos', 'zRotation', 'decodedId']
)

def createAllTables(c):

	c.execute('''DROP TABLE IF EXISTS FRAME_CONTAINER''')
	c.execute('''DROP TABLE IF EXISTS FRAME''')
	c.execute('''DROP TABLE IF EXISTS DETECTIONS''')

	c.execute('''CREATE TABLE FRAME_CONTAINER (\
	FC_ID  INT PRIMARY KEY,\
	ID  TEXT,\
	CAM_ID  INT,\
	FROM_TS  INT,\
	TO_TS  INT\
	);''')

	c.execute('''CREATE TABLE FRAME (\
	FRAME_ID  INT PRIMARY KEY,\
	FC_ID  INT,\
	TIMESTAMP  INT,\
	FOREIGN KEY(FC_ID) REFERENCES FRAME_CONTAINER(FC_ID)\
	);''')

	c.execute('''CREATE TABLE DETECTIONS (\
	FRAME_ID  INT,\
	XPOS  INT,\
	YPOS  INT,\
	ZROTATION FLOAT,\
	ID INT,\
	FOREIGN KEY(FRAME_ID) REFERENCES FRAME(FRAME_ID)\
	);''')


def run(path_to_db, path_to_repo, conf, start_string, time_delta):

	db_path = path_to_db
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	createAllTables(c)

	repo = Repository(path_to_repo)
	confidence = conf

	start = start_string
	start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
	start_ts = start_dt.timestamp()

	end_dt = start_dt + datetime.timedelta(hours=time_delta)
	end_ts = end_dt.timestamp()

	files = list(repo.iter_fnames(begin=start_ts, end=end_ts))
	print("Number of files: {}".format(len(files)))

	# ADD ALL THE STUFF TO THE DB
	#############################
	my_fc_id = 0
	my_frame_id = 0

	# alle dateien bzw. FrameConatiner interieren
	for file in files:
		print("Progess: {}/{}".format(my_fc_id+1,len(files)))
		fc = load_frame_container(file)

		# pro Framecontainer ein Eintrag in die FrameContainer Table machen
		c.execute("insert into frame_container (fc_id, id, cam_id, from_ts, to_ts) values (?, ?, ?, ?, ?)",
			(my_fc_id, str(fc.id), fc.camId, fc.fromTimestamp, fc.toTimestamp))


		# alle Frames iterieren
		tpls = []

		for f in fc.frames:
			# pro frame einen Eintrag in Frame Tabelle machen
			c.execute("insert into frame (frame_id, fc_id, timestamp) values (?, ?, ?)",
				(my_frame_id, my_fc_id, f.timestamp))

			# alle Detections iterieren
			for d in f.detectionsUnion.detectionsDP:
				d = Detection(my_frame_id, d.xpos, d.ypos, d.zRotation, list(d.decodedId))
				tpls.append(d)
	        
			# hochzaehlen
			my_frame_id += 1

		df = pd.DataFrame(tpls) 
		df = prep.calcIds(df, confidence)
		df.drop('confidence', axis=1, inplace=True)

		# Detections zu db hinzufuegen
		df.to_sql('DETECTIONS', conn, if_exists='append', index=False)

		# hochzaehlen!
		my_fc_id += 1


	conn.commit()
	conn.close()


if __name__ == '__main__':

	# path_to_db, path_to_repo, conf, start_string, time_delta
	if (len(sys.argv) == 6 ):
		path_to_db = sys.argv[1]
		path_to_repo = sys.argv[2]

		conf = float(sys.argv[3])

		start_string = str(sys.argv[4])
		time_delta = int(sys.argv[5])

		run(path_to_db, path_to_repo, conf, start_string, time_delta)
	else:
		print("Usage:\npython3 bbbinary_to_db.py <path_to_db> <path_to_repo> <confidence> <start_time_string> <time_delta_hours>")
		print("Example:\npython3 bbbinary_to_db.py '/storage/mi/aschle/data.db' '/storage/mi/aschle/days/' 0.99 2015-08-21T00:00:00Z 1")


# "/storage/mi/aschle/data.db"
# "/storage/mi/aschle/days/"
# 0.99
# "2015-08-21T00:00:00Z"
# 24