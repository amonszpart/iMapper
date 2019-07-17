######
# Actor 0
######

# charness: 0.493107
D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/222_242";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 222 242 --batch-size 10 --dest-dir opt2 --candidates opt1/222_242 --n-candidates 200 -tc 0.30
fi

# charness: 0.483465
D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/300_320";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 300 320 --batch-size 10 --dest-dir opt2 --candidates opt1/300_320 --n-candidates 200 -tc 0.30
fi

# charness: 0.470942
D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/088_108";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 088 108 --batch-size 10 --dest-dir opt2 --candidates opt1/088_108 --n-candidates 200 -tc 0.30
fi

# charness: 0.337767
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/243_263";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 243 263 --batch-size 10 --dest-dir opt2 --candidates opt1/243_263 --n-candidates 200 -tc 0.30
# fi

# charness: 0.246848
D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/168_188";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 168 188 --batch-size 10 --dest-dir opt2 -tc 0.30
fi

# charness: 0.25448
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/350_370";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 350 370 --batch-size 10 --dest-dir opt2 --candidates opt1/350_370 --n-candidates 200 -tc 0.30
# fi

# charness: 0.192039
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/279_299";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 279 299 --batch-size 10 --dest-dir opt2 --candidates opt1/279_299 --n-candidates 200 -tc 0.30
# fi

# charness: 0.259548
D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/401_421";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 401 421 --batch-size 10 --dest-dir opt2 -tc 0.30
fi

# charness: 0.244682
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/198_218";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 198 218 --batch-size 10 --dest-dir opt2 --candidates opt1/198_218 --n-candidates 200 -tc 0.30
# fi

# charness: 0.219236
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/485_505";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 485 505 --batch-size 10 --dest-dir opt2 --candidates opt1/485_505 --n-candidates 200 -tc 0.30
# fi

# charness: 0.25596
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/325_345";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 325 345 --batch-size 10 --dest-dir opt2 --candidates opt1/325_345 --n-candidates 200 -tc 0.30
# fi

# charness: 0.187715
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/067_087";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 067 087 --batch-size 10 --dest-dir opt2 --candidates opt1/067_087 --n-candidates 200 -tc 0.30
# fi

# charness: 0.178733
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/109_129";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 109 129 --batch-size 10 --dest-dir opt2 --candidates opt1/109_129 --n-candidates 200 -tc 0.30
# fi

# charness: 0.188702
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/375_395";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 375 395 --batch-size 10 --dest-dir opt2 --candidates opt1/375_395 --n-candidates 200 -tc 0.30
# fi

# charness: 0.182241
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/146_166";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 146 166 --batch-size 10 --dest-dir opt2 --candidates opt1/146_166 --n-candidates 200 -tc 0.30
# fi

# charness: 0.220937
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/034_054";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 034 054 --batch-size 10 --dest-dir opt2 --candidates opt1/034_054 --n-candidates 200 -tc 0.30
# fi

# charness: 0.184301
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/506_526";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 506 526 --batch-size 10 --dest-dir opt2 --candidates opt1/506_526 --n-candidates 200 -tc 0.30
# fi

# charness: 0.181288
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/002_022";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 002 022 --batch-size 10 --dest-dir opt2 --candidates opt1/002_022 --n-candidates 200 -tc 0.30
# fi

# charness: 0.175677
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/432_452";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 432 452 --batch-size 10 --dest-dir opt2 --candidates opt1/432_452 --n-candidates 200 -tc 0.30
# fi

# charness: 0.175291
# D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/455_475";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 455 475 --batch-size 10 --dest-dir opt2 --candidates opt1/455_475 --n-candidates 200 -tc 0.30
# fi

D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/247_248";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 247 248 --batch-size 10 --dest-dir opt2 --candidates opt1/237_257 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/253_258";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 253 258 --batch-size 10 --dest-dir opt2 --candidates opt1/245_265 --n-candidates 200 -tc 0.30 --remove-objects
fi

D="/media/data/amonszpa/stealth/shared/video_recordings/library3-tog/opt2b/376_377";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/library3-tog/skel_library3-tog_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 376 377 --batch-size 10 --dest-dir opt2 --candidates opt1/366_386 --n-candidates 200 -tc 0.30 --remove-objects
fi



