# iMapper: Interaction-guided Scene Mapping from Monocular Videos [(link)](http://geometry.cs.ucl.ac.uk/projects/2019/imapper/)

![teaser](http://geometry.cs.ucl.ac.uk/projects/2019/imapper/paper_docs/teaser.jpg)

## i3DB

### Download videos and images
    sh get_data.sh

### Open a ground truth scenelet

#### Setup
    conda create --name iMapper python=3 numpy -y

#### Usage
    conda activate iMapper
    export PYTHONPATH=$(pwd); python3 ./example.py
    
## Run iMapper on a video

### Requirements:
* docker
* nvidia-docker
* CUDA capable GPU

Have a folder containing 
* a video (e.g., `video.mp4`) and 
* camera intrinsics
in `intrinsics.json` e.g., `[[1920.0, 0.0, 960.0], [0.0, 1920.0, 540.0], [0.0, 0.0, 1.0]]`
Replace the variables `PATH_TO_FOLDER_CONTAINING_VIDEO` and `VIDEO` below
Adjust GPU id if needed.

```shell
git clone https://github.com/amonszpart/iMapper.git
cd iMapper
docker build -t iMapper imapper/docker
nvidia-docker run -it --name iMapper \
 -v ${PATH_TO_FOLDER_CONTAINING_VIDEO}:/data:rw \
 iMapper \
 CUDA_VISIBLE_DEVICES=0 python3 run_video.py /data/${VIDEO} --gpu-id 0
```

