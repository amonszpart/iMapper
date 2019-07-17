######
# Actor 0
######

# charness: 0.509933
D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/023_043";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 023 043 --batch-size 10 --dest-dir opt2 --candidates opt1/023_043 --n-candidates 200 -tc 0.30
fi

# charness: 0.534372
D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/186_206";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 186 206 --batch-size 10 --dest-dir opt2 --candidates opt1/186_206 --n-candidates 200 -tc 0.30
fi

# charness: 0.397003
D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/131_151";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 131 151 --batch-size 10 --dest-dir opt2 --candidates opt1/131_151 --n-candidates 200 -tc 0.30
fi

# charness: 0.351639
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/300_320";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 300 320 --batch-size 10 --dest-dir opt2 --candidates opt1/300_320 --n-candidates 200 -tc 0.30
# fi

# charness: 0.404783
D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/066_086";
if [ ! -d ${D} ]; then
	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 066 086 --batch-size 10 --dest-dir opt2 -tc 0.30
fi

# charness: 0.26201
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/045_065";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 045 065 --batch-size 10 --dest-dir opt2 --candidates opt1/045_065 --n-candidates 200 -tc 0.30
# fi

# charness: 0.215049
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/247_267";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 247 267 --batch-size 10 --dest-dir opt2 --candidates opt1/247_267 --n-candidates 200 -tc 0.30
# fi

# charness: 0.180649
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/087_107";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 087 107 --batch-size 10 --dest-dir opt2 --candidates opt1/087_107 --n-candidates 200 -tc 0.30
# fi

# charness: 0.189391
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/212_232";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 212 232 --batch-size 10 --dest-dir opt2 --candidates opt1/212_232 --n-candidates 200 -tc 0.30
# fi

# charness: 0.187352
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/153_173";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 153 173 --batch-size 10 --dest-dir opt2 --candidates opt1/153_173 --n-candidates 200 -tc 0.30
# fi

# charness: 0.185144
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/002_022";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 002 022 --batch-size 10 --dest-dir opt2 --candidates opt1/002_022 --n-candidates 200 -tc 0.30
# fi

# charness: 0.186065
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/108_128";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 108 128 --batch-size 10 --dest-dir opt2 --candidates opt1/108_128 --n-candidates 200 -tc 0.30
# fi

# charness: 0.185095
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/279_299";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 279 299 --batch-size 10 --dest-dir opt2 --candidates opt1/279_299 --n-candidates 200 -tc 0.30
# fi

# charness: 0.184467
# D="/media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/opt2b/328_348";
# if [ ! -d ${D} ]; then
# 	python3 stealth/pose/opt_consistent.py -silent --wp 1 --ws 0.05 --wi 1. --wo 1. -w-occlusion -w-static-occlusion --maxiter 15 --nomocap -v /media/data/amonszpa/stealth/shared/video_recordings/office1-1-tog-lcrnet/skel_office1-1-tog-lcrnet_unannot.json independent -s /media/data/amonszpa/stealth/shared/pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling --gap 328 348 --batch-size 10 --dest-dir opt2 --candidates opt1/328_348 --n-candidates 200 -tc 0.30
# fi



