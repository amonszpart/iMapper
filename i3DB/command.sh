if [ $# -ne 3 ];
then
    echo "Usage: command.sh <avconvrate> <start> <duration>"
    exit 1
fi

rate=$1
start=$2
duration=$3

scene=$(basename $(pwd))
echo "scene: $scene"
video=$(find . -maxdepth 1 -name '*.MOV' -o -name '*.mp4' -o -name '*.MP4' -o -name '*.m4v' -o -name '*.webm' -o -name '*.avi') && echo "Video is $video"
rm -r origjpg
mkdir origjpg
#cmd="ffmpeg -i $video -ss ${start} -t ${duration} -q:v 2 -r $rate origjpg/color_%05d.jpg"
#echo $cmd

avconv -i "${video}" -ss ${start} -t ${duration} -q:v 2 -r $rate origjpg/color_%05d.jpg
rate2=$(python -c "print(max(24./${rate}, 1.))")
data="{\n\
   \t\"rate\": ${rate2},\n\
   \t\"rate-avconv\": ${rate},\n\
   \t\"start:\": \"${start}\",\n\
   \t\"duration:\": \"${duration}\"\n\
}\n"
echo -e "$data"
echo "$data">video_params.json

mkdir input
#ffmpeg -i "$video" -r 24 -ss ${start} -t ${duration} -an input/input_cropped.mp4
if [ ! -f input/input_cropped_10fps.mp4 ]; then
    echo "ffmpeg -i \"$video\" -r 10 -ss ${start} -t ${duration} -an input/input_cropped_10fps.mp4"
    #ffmpeg -i "$video" -r 10 -ss ${start} -t ${duration} -an input/input_cropped_10fps.mp4
fi
