######
# Actor 0
######

# charness: 0.5327
D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/110_130";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 110 130 --batch-size 10 --dest-dir opt2 --candidates opt1/110_130 --n-candidates 200 -tc 0.30
fi

# charness: 0.439639
D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/033_053";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 033 053 --batch-size 10 --dest-dir opt2 --candidates opt1/033_053 --n-candidates 200 -tc 0.30
fi

# charness: 0.363552
D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/158_178";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 158 178 --batch-size 10 --dest-dir opt2 --candidates opt1/158_178 --n-candidates 200 -tc 0.30
fi

# charness: 0.290531
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/179_199";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 179 199 --batch-size 10 --dest-dir opt2 --candidates opt1/179_199 --n-candidates 200 -tc 0.30
# fi

# charness: 0.193588
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/089_109";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 089 109 --batch-size 10 --dest-dir opt2 --candidates opt1/089_109 --n-candidates 200 -tc 0.30
# fi

# charness: 0.207317
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/012_032";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 012 032 --batch-size 10 --dest-dir opt2 --candidates opt1/012_032 --n-candidates 200 -tc 0.30
# fi

# charness: 0.18197
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/200_220";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 200 220 --batch-size 10 --dest-dir opt2 --candidates opt1/200_220 --n-candidates 200 -tc 0.30
# fi

# charness: 0.198412
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/054_074";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 054 074 --batch-size 10 --dest-dir opt2 --candidates opt1/054_074 --n-candidates 200 -tc 0.30
# fi

# charness: 0.178938
# D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/131_151";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 131 151 --batch-size 10 --dest-dir opt2 --candidates opt1/131_151 --n-candidates 200 -tc 0.30
# fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/014_017";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 014 017 --batch-size 10 --dest-dir opt2 --candidates opt1/005_025 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/019_022";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 019 022 --batch-size 10 --dest-dir opt2 --candidates opt1/010_030 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/065_067";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 065 067 --batch-size 10 --dest-dir opt2 --candidates opt1/056_076 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/083_086";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 083 086 --batch-size 10 --dest-dir opt2 --candidates opt1/074_094 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/093_094";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 093 094 --batch-size 10 --dest-dir opt2 --candidates opt1/083_103 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/108_109";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 108 109 --batch-size 10 --dest-dir opt2 --candidates opt1/098_118 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/131_134";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 131 134 --batch-size 10 --dest-dir opt2 --candidates opt1/122_142 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/144_147";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 144 147 --batch-size 10 --dest-dir opt2 --candidates opt1/135_155 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/155_157";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 155 157 --batch-size 10 --dest-dir opt2 --candidates opt1/146_166 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/188_190";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 188 190 --batch-size 10 --dest-dir opt2 --candidates opt1/179_199 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/opt2b/205_207";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 10. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/skel_lobby22-1-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 205 207 --batch-size 10 --dest-dir opt2 --candidates opt1/196_216 --n-candidates 200 -tc 0.30 --remove-objects
fi



