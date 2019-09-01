import argparse
import glob
import json
import os
import sys
from subprocess import check_call


def _parse_args(argv):
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        'f', type=str,
        help='Video file')
    parser.add_argument(
        '--rate', type=int,
        help='Video subsample rate',
        default=10)
    parser.add_argument(
        '--path-imapper', type=str,
        help='Path to imapper python package',
        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        '--path-lfd', type=str,
        help='Path to Lifting from the Deep root',
        default='/opt/Lifting-from-the-Deep-release')
    parser.add_argument(
        '--gpu-id', type=str,
        help='CUDA_VISIBLE_DEVICES=<GPU_ID>',
        default='0'
    )
    args = parser.parse_args(argv)
    
    return args


# "pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed_withmocap_ct_full_sampling.tar.gz"


def run_tome3d(path_poses):
    "main_denis.py s8 -d /media/data/amonszpa/stealth/shared/video_recordings/lobby22-1-tog/denis -smooth 10"


def create_origjpg(path_scene, args):
    p_origjpg = os.path.join(path_scene, 'origjpg')
    if os.path.isdir(p_origjpg) and len(
      list(glob.glob('{}/*.jpg'.format(p_origjpg)))):
        print('[create_origjpg] origjpg exists, not rerunning')
        return
    p_command_sh = os.path.normpath(
        os.path.join(args.path_imapper, 'i3DB', 'command.sh'))
    assert os.path.isfile(p_command_sh), \
        "Can't find command.sh at {}".format(p_command_sh)
    
    cmd = '/usr/bin/env bash {:s} {:d}'.format(p_command_sh, args.rate)
    check_call(cmd.split(' '), cwd=path_scene)
    
    p_scene_params = os.path.join(path_scene, 'scene_params.json')
    if not os.path.isfile(p_scene_params):
        json.dump(
            {"depth_init": 10.0, "actors": 1, "ground_rot": [0.0, 0.0, 0.0]},
            open(p_scene_params, 'w'))


def run_lfd(path_scene, args):
    p_denis = os.path.join(path_scene, 'denis')
    if os.path.isdir(p_denis) and len(
      list(glob.glob('{}/*.json'.format(p_denis)))):
        print('[run_lfd] LFD skeletons exist, not rerunning')
        return
    
    p_origjpg = os.path.join(path_scene, 'origjpg')
    if not os.path.isdir(p_origjpg) or not len(
      list(glob.glob('{}/*.jpg'.format(p_origjpg)))):
        print('[run_lfd] origjpg does not exists, can\'t run this step')
        return
    
    cmd = 'python2 {:s}/demo_aron.py -d {:s} --no-vis' \
        .format(args.path_lfd, p_origjpg)
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '{}:{}'.format(my_env['PYTHONPATH'], args.path_lfd)
    my_env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    check_call(cmd.split(' '), cwd=args.path_lfd, env=my_env)


def main(argv):
    args = _parse_args(argv)
    assert os.path.isfile(args.f), \
        'Video does not exist: {}'.format(args.f)
    print('Working with {}'.format(args.f))
    
    path_scene = os.path.dirname(args.f)
    create_origjpg(path_scene=path_scene, args=args)
    
    run_lfd(path_scene=path_scene, args=args)


if __name__ == '__main__':
    main(sys.argv[1:])
