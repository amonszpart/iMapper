# iMapper: Interaction-guided Scene Mapping from Monocular Videos [(link)](http://geometry.cs.ucl.ac.uk/projects/2019/imapper/)

![teaser](http://geometry.cs.ucl.ac.uk/projects/2019/imapper/paper_docs/teaser.jpg)

### Download videos and images
    sh get_data.sh

### Open a ground truth scenelet

#### Setup
    conda create --name iMapper python=3 numpy -y

#### Usage
    conda activate iMapper
    export PYTHONPATH=$(pwd); python3 ./example.py
