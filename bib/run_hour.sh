for i in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23"

	do
	echo $i
	python3 pipeline_frames_parallel.py /storage/mi/aschle/1hour/20150821-$i/ 0.99 160 6 12 20150821-$i-1h-allCams

done
