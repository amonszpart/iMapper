rate=10
start="00:00:03.750"
duration="00:00:25.500"
../command.sh $rate $start $duration

# scene=$(basename $(pwd))
# echo "scene: $scene"
# video=$(find . -name '*.MOV' -o -name '*.mp4' -o -name '*.MP4') && echo "Video is $video"
# rm -r origjpg
# mkdir origjpg
# ffmpeg -i $video -ss ${start} -t ${duration} -q:v 2 -r $rate origjpg/color_%05d.jpg
# chown 1000:18429 -R origjpg
# rate2=$(python -c "print(max(24./${rate}, 1.))")
# echo -e "{\n\t\"rate\": ${rate2}\n}" >video_params.json
# chown 1000:18429 -R video_params.json

#cd /data/data/scenes/$scene
#nvidia-docker start arons-tf-denis
#nvidia-docker attach arons-tf-denis

#cd /data/code/Lifting-from-the-Deep-release/
#./demo_aron.py -d ../../data/scenes/$scene/origjpg/ --no-vis --vis-thresh 5e-4
