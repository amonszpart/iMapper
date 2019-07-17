"""
Setup:
    conda create --name iMapper python=3 numpy -y \
    conda activate iMapper

Usage:
    export PYTHONPATH=$(pwd); python3 example.py
"""


from imapper.logic.scenelet import Scenelet

if __name__ == '__main__':
    scenelet = Scenelet.load('i3DB/Scene04/gt/skel_lobby19-3_GT.json')
    skeleton = scenelet.skeleton
    print('Have {} poses and {} objects'
          .format(len(skeleton.get_frames()), len(scenelet.objects)))