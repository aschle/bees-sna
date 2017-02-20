for i in {20..26}

	do
	echo "$i"
	python3 pipeline_frames_parallel.py /mnt/data2016/data/ 2016-07-${i}T00:00:00Z 1440 0.99 160 6 8 2015-08-${i}T00:00:00Z_1d_allCams 2016

done
