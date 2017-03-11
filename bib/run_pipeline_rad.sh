#!/bin/bash
####################################
#
# Run pipeline with different params
#
####################################

# interaction length
for i in 238 265 292 318 189 159 133 106
	do
		echo $i
		python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-13T08:00:00Z 600 95 $i 3 8 0813_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-14T08:00:00Z 600 95 $i 3 8 0814_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-16T08:00:00Z 600 95 $i 3 8 0816_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-17T08:00:00Z 600 95 $i 3 8 0817_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-20T08:00:00Z 600 95 $i 3 8 0820_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-22T08:00:00Z 600 95 $i 3 8 0822_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-24T08:00:00Z 600 95 $i 3 8 0824_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-08-25T08:00:00Z 600 95 $i 3 8 0825_10h 2016 2
                python3 pipeline_frames_parallel.py /mnt/storage/beesbook/repo_season_2016_fixed 2016-09-02T08:00:00Z 600 95 $i 3 8 0902_10h 2016 2
	done
