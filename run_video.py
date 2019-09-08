import argparse
import glob
import json
import os
import shutil
import sys
from itertools import product
from subprocess import check_call


def _parse_args(argv):
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        'f', type=str,
        help='Video file')
    parser.add_argument(
        '--gpu-id', type=str,
        help='CUDA_VISIBLE_DEVICES=<GPU_ID>'
    )
    parser.add_argument(
        '--max-candidate-rank', type=int,
        help='Which candidate to still consider for a solution. '
             '1: only one, 3: first three, etc.',
        default=2
    )
    parser.add_argument(
        '--rate', type=int,
        help='Video subsample rate',
        default=10)
    parser.add_argument(
        '--path-imapper', type=str,
        help='Path to iMapper, containing the python package \'imapper\'',
        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        '--path-lfd', type=str,
        help='Path to Lifting from the Deep root',
        default='/opt/Lifting-from-the-Deep-release')
    parser.add_argument(
        '--span-size', type=int,
        help='Number of frames to explain with one scenelet',
        default=20
    )
    args = parser.parse_args(argv)
    
    return args


def create_origjpg(path_scene, args):
    p_origjpg = os.path.join(path_scene, 'origjpg')
    if os.path.isdir(p_origjpg) and len(
      list(glob.glob('{}/*.jpg'.format(p_origjpg)))):
        print('[create_origjpg]\torigjpg exists, not rerunning')
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
        print('[run_lfd]\t\t\tLFD skeletons exist, not rerunning')
        return p_denis
    
    p_origjpg = os.path.join(path_scene, 'origjpg')
    if not os.path.isdir(p_origjpg) or not len(
      list(glob.glob('{}/*.jpg'.format(p_origjpg)))):
        print('[run_lfd]\torigjpg does not exists, can\'t run this step')
        return None
    
    cmd = 'python2 {:s}/demo_aron.py -d {:s} --no-vis' \
        .format(args.path_lfd, p_origjpg)
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '{}:{}:{}'.format(args.path_lfd, args.path_imapper,
                                             my_env['PYTHONPATH'])
    my_env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    check_call(cmd.split(' '), cwd=args.path_lfd, env=my_env)
    
    return p_denis


def run_tome3d(path_poses, args):
    from imapper.pose.main_denis import main as main_denis
    
    path_scene = os.path.dirname(path_poses)
    name_scene = os.path.basename(path_scene)
    p_skeleton = os.path.join(path_scene,
                              'skel_{}_unannot.json'.format(name_scene))
    if os.path.isfile(p_skeleton):
        print('[run_tome3d]\t\tSkeleton file already exists,'
              ' not rerunning: {}'.format(p_skeleton))
        return p_skeleton
    
    p_intrinsics = os.path.normpath(
        os.path.join(os.path.dirname(path_poses), 'intrinsics.json'))
    if not os.path.isfile(p_intrinsics):
        raise IOError(
            'Could not find intrinsics at {}\n'
            'Example content: '
            '\t[[1955.46, 0.0, 960.33], '
            '[0.0, 1953.0, 540.0], '
            '[0.0, 0.0, 1.0]]'.format(p_intrinsics))
    _argv = ['mycamera', '-d', path_poses, '-smooth', '10']
    if os.environ.get('BLENDER') is None:
        os.environ['BLENDER'] = '/usr/bin/blender'
    if not os.path.isfile(os.environ['BLENDER']):
        raise RuntimeError(
            'Could not find blender at {}'.format(os.environ['BLENDER']))
    main_denis(_argv)


def fit_full_scene(path_scene, args):
    from imapper.pose.fit_full_video import \
        main as main_fit_full_video, extract_gaps
    
    p_scenelet_db = os.path.normpath(os.path.join(
        path_scene, os.pardir,
        'pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed'
        '_withmocap_ct_full_sampling'))
    if not os.path.isdir(p_scenelet_db):
        raise RuntimeError(
            'Could not find directory {}, please download it from '
            'http://geometry.cs.ucl.ac.uk/projects/2019/imapper/'
            'pigraph_scenelets__linterval_squarehist_large_radiusx2_smoothed'
            '_withmocap_ct_full_sampling.tar.gz'.format(p_scenelet_db))
    p_opt1 = os.path.normpath(os.path.join(path_scene, 'opt1'))
    if not os.path.isdir(p_opt1):
        _argv = ['--wp', '1',  # projection term
                 '--ws', '0.5',  # smoothness term
                 '-no-isec',  # no intersection term
                 '--maxiter', '15',  # lbfgs iterations
                 path_scene,  # path to scene
                 '1',  # start frame
                 '{:d}'.format(args.span_size),  #  gap width
                 '-s', p_scenelet_db,  # scenelet database
                 '--batch-size', '1500',  #  scenelet fit batch size
                 '--output-n', '200'  #  how many fits to keep
                 ]
        main_fit_full_video(_argv)
    else:
        print('[fit_full_scene] Not running, already exists: {}'
              .format(p_opt1))
    
    p_opt2b = os.path.normpath(os.path.join(path_scene, 'opt2b'))
    if not os.path.isdir(p_opt2b):
        extract_gaps([path_scene, p_scenelet_db])
        
        p_gap_command = os.path.join(path_scene, 'gap_command_new.sh')
        if not os.path.isfile(p_gap_command):
            raise IOError(
                'Could not find output of step: {}'.format(p_gap_command))
        
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '{}:{}:{}' \
            .format(args.path_lfd, args.path_imapper, my_env['PYTHONPATH'])
        my_env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        check_call('/usr/bin/env bash {}'.format(p_gap_command).split(' '),
                   env=my_env, cwd=args.path_imapper)
    else:
        print('[fit_full_scene] Not running, already exists: {}'
              .format(p_opt2b))


def combine_candidates(path_scene, args):
    p_opt3_input = os.path.join(path_scene, 'opt3_input')
    if os.path.isdir(p_opt3_input):
        print('[combine_candidates] Not running, already exists: {}'.format(
            p_opt3_input))
        return
    
    p_opt2b = os.path.normpath(os.path.join(path_scene, 'opt2b'))
    if not os.path.isdir(p_opt2b):
        raise IOError('Could not find {}'.format(p_opt2b))
    spans = (os.path.join(p_opt2b, d)
             for d in os.listdir(p_opt2b)
             if os.path.isdir(os.path.join(p_opt2b, d)))
    
    def scn_start_end(path):
        return [int(part) for part in os.path.basename(path).split('_')]
    
    def scn_len(path):
        parts = scn_start_end(path)
        return parts[1] - parts[0]
    
    def candidate_rank(path):
        return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
    
    entries = [
        sorted(os.path.join(span, d)
               for d in os.listdir(span)
               if os.path.isfile(os.path.join(span, d))
               and d.startswith('skel')
               and (
                 ((scn_len(span) == args.span_size)
                  and (candidate_rank(d) < args.max_candidate_rank))
                 or ((scn_len(span) != args.span_size)
                     and (candidate_rank(d) == 0))
               )
               )
        for span in spans]
    gen = product(*entries)
    os.makedirs(p_opt3_input, exist_ok=True)
    for elements in gen:
        p_comb = \
            os.path.join(
                p_opt3_input,
                'c{}'.format(''.join('_{:02d}'.format(candidate_rank(element))
                                     for element in elements)))
        if os.path.exists(p_comb):
            shutil.rmtree(p_comb)
        os.makedirs(p_comb)
        for element in elements:
            p_scene = os.path.basename(element)
            parts, ext = os.path.splitext(p_scene)
            parts = parts.split('_')
            
            p_dest = os.path.join(p_comb, element.split(os.sep)[-2])
            if not os.path.isdir(p_dest):
                os.makedirs(p_dest)
            p_dest2 = os.path.join(p_dest,
                                   '{}{}'.format('_'.join(parts[:-1]), ext))
            shutil.copy2(element, p_dest2)
            print('[combine candidates] Copied\n\t{:s} to\n\t{:s}'
                  .format(element, p_dest2))
            
            p_objects = "%s_objects" % os.path.splitext(element)[0]
            if os.path.isdir(p_objects):
                shutil.copytree(
                    p_objects,
                    os.path.join(p_dest, os.path.basename(p_objects)))
                print('[combine candidates] Copied\n\t{:s} to\n\t{:s}'
                      .format(p_objects,
                              os.path.join(p_dest,
                                           os.path.basename(p_objects))))


def optimize_combinations(path_scene, args):
    from imapper.pose.opt_consistent import main as main_opt
    from imapper.blender.show_scene import main as main_show
    
    p_opt3 = os.path.normpath(os.path.join(path_scene, 'opt3_input'))
    if not os.path.isdir(p_opt3):
        raise IOError('Could not find {}'.format(p_opt3))
    combinations = sorted(
        d for d in glob.glob('{}/c_*'.format(p_opt3))
        if os.path.isdir(d))
    p_outputs = os.path.join(path_scene, 'outputs')
    for path_input in combinations:
        _argv = [
            '--wp', '1',  #  projection term
            '--ws', '0.05',  # smoothness term
            '--wo', '1',  # occlusion term
            
            '--wi', '20',  # object-object intersection term
            # Note: the joint-object intersection term is `:=0.5*wo` in
            # the constructor of imapper.pose.opt_consistent.Weights
            
            '-w-occlusion',  # use occlusion terms
            '-w-static-occlusion',  # objects can occlude non-replaced poses
            '-v', path_scene,  #  path to video
            '--d-pad', '0.05',  # spatial padding around objects (in m)
            'consistent',  #  flag for final optimization
            #  source scenelets
            '-input', os.path.relpath(path_input, path_scene)
        ]
        # main_opt(_argv)
        p_output = os.path.join(path_scene, 'output')
        combination_id = '_'.join(os.path.basename(path_input).split('_')[1:])
        p_output2 = os.path.join(p_outputs, 'output_{}'.format(combination_id))
        # shutil.move(p_output, p_output2)
        print('Moved {} to {}'.format(p_output, p_output2))
        
        _argv = [
            '--blender', '/usr/bin/blender',
            '--scene', os.path.join(p_output2, 'skel_output.json'),
            '--video', '',
            '--fps', '24',
            '--out', os.path.join(p_output2, 'skel_output.blend'),
            '--quick', '-b'
        ]
        main_show(_argv)


def main(argv):
    args = _parse_args(argv)
    assert os.path.isfile(args.f), \
        'Video does not exist: {}'.format(args.f)
    print('[main] Working with {}'.format(args.f))
    
    path_scene = os.path.dirname(args.f)
    
    create_origjpg(path_scene=path_scene, args=args)

    path_poses = run_lfd(path_scene=path_scene, args=args)
    run_tome3d(path_poses, args)

    fit_full_scene(path_scene=path_scene, args=args)
    
    combine_candidates(path_scene=path_scene, args=args)
    
    optimize_combinations(path_scene=path_scene, args=args)


if __name__ == '__main__':
    main(sys.argv[1:])
