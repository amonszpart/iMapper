if [ $# -lt 1 ];
then
    echo "Usage: command.sh <avconvrate> [<start> <duration>]"
    exit 1
fi

rate="$1"
if [ -n "$2" ]; then start="$2"; fi
if [ -n "$3" ]; then duration="$3"; fi

scene="$(basename $(pwd))"
echo "Scene name is: $scene"

video=$(find . -maxdepth 1 -name '*.MOV' -o -name '*.mp4' -o -name '*.MP4' -o -name '*.m4v' -o -name '*.webm' -o -name '*.avi')
echo "Video is $video"

if [ -d "origjpg" ]; then
  rm -r origjpg
fi
mkdir origjpg

if [ -n "${start}" ]; then _START="-ss ${start}"; fi
if [ -n "${duration}" ]; then _DURATION="-t ${duration}"; fi

avconv -i "${video}" ${_START} ${_DURATION} -q:v 2 -r ${rate} origjpg/color_%05d.jpg

rate2=$(python -c "print(max(24./${rate}, 1.))")
data="{\n\
   \t\"rate\": ${rate2},\n\
   \t\"rate-avconv\": ${rate},\n\
   \t\"start:\": \"${start}\",\n\
   \t\"duration:\": \"${duration}\"\n\
}\n"
echo -e "$data"
printf "$data">video_params.json

test input || mkdir input
if [ ! -f input/input_cropped_10fps.mp4 ]; then
    echo "avconv -i \"$video\" -r 10 -ss ${start} -t ${duration} -an input/input_cropped_10fps.mp4"
fi
