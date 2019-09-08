import argparse
import os
import sys
import pdb
import bdb
import traceback

import numpy as np

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
      os.path.realpath(__file__)), '../../')))

from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.util.my_argparse import argparse_check_exists
from imapper.util.stealth_logging import lg
from imapper.pose.visualization.show_charness_scores import show_folder
from imapper.pose.visualization.extract_gaps import main as extract_gaps
from imapper.util.json import json


def main(argv):
    from imapper.pose.opt_consistent import main as opt_consistent
    pjoin = os.path.join
    parser = argparse.ArgumentParser("Fit full video")
    parser.add_argument("video", type=argparse_check_exists,
                        help="Input path")
    parser.add_argument("step_size", type=int,
                        help="Stepsize in frames.")
    parser.add_argument("window_size", type=int,
                        help="Window size in frames.")
    parser.add_argument(
      '--wp', type=float, help="Projection weight.", default=1.
    )
    parser.add_argument(
      '--ws', type=float, help="Smoothness weight.", default=.1
    )
    parser.add_argument(
      '--wo', type=float, help="Occlusion weight.",
      default=0.1
    )
    parser.add_argument(
      '--wi', type=float, help="Intersection weight.", default=1.
      # used to be 10.
    )
    parser.add_argument(
      '--gtol', type=float, help="Optimizer gradient tolerance (termination "
                                 "criterion).", default=1e-6
    )
    parser.add_argument(
      '--maxiter', type=int, help="Optimizer max number of iterations.",
      default=0
    )
    parser.add_argument('-w-occlusion', action='store_true',
                        help="Estimate occlusion score.")
    parser.add_argument('-no-isec', action='store_true',
                        help='Don\'t use intersection terms')
    parser.add_argument('--dest-dir', type=str,
                        help="Name of subdirectory to save output to.",
                        default='opt1')
    parser.add_argument("-s", "--d-scenelets",
                        dest='s', type=argparse_check_exists,
                        help="Folder containing original PiGraphs scenelets")
    parser.add_argument('--batch-size', type=int,
                        help="How many scenelets to optimize at once.",
                        default=1500)
    parser.add_argument('--output-n', type=int,
                        help="How many candidates to output per batch and "
                             "overall.", default=200)
    parser.add_argument('--filter-same-scene', action='store_true',
                        help="Hold out same scene scenelets.")
    args = parser.parse_args(argv)

    # get video parent directory
    d_query = args.video if os.path.isdir(args.video) \
        else os.path.dirname(args.video)

    # save call log to video directory
    with open(pjoin(d_query, 'args_opt_consistent.txt'), 'a') as f_args:
        f_args.write('(python3 ')
        f_args.write(" ".join(sys.argv))
        f_args.write(")\n")

    # parse video path
    name_query = os.path.split(d_query)[-1]
    p_query = pjoin(d_query, "skel_%s_unannot.json" % name_query) \
        if os.path.isdir(args.video) else args.video
    assert p_query.endswith('.json'), "Need a skeleton file"
    print("name_query: %s" % name_query)

    cache_scenes = None

    skipped = []
    # load initial video path (local poses)
    query = Scenelet.load(p_query, no_obj=True)
    frame_ids = query.skeleton.get_frames()
    half_window_size = args.window_size // 2
    for mid_frame_id in range(frame_ids[0] + half_window_size,
                              frame_ids[-1] - half_window_size + 1,
                              args.step_size):
        gap = (mid_frame_id - half_window_size,
               mid_frame_id + half_window_size)
        assert gap[0] >= frame_ids[0]
        assert gap[1] <= frame_ids[-1]
        pose_count = sum(1 for _frame_id in range(gap[0], gap[1]+1)
                         if query.skeleton.has_pose(_frame_id))
        if pose_count < 9:
            print("Skipping span because not enough poses: %s" % pose_count)
            skipped.append((gap, pose_count))
        same_actor = query.skeleton.n_actors == 1  # type: bool
        if not same_actor:
            same_actor = query.skeleton.get_actor_id(frame_id=gap[0]) \
                         == query.skeleton.get_actor_id(frame_id=gap[1])
        if not same_actor:
            print('skipping gap {:d}...{:d}, not same actor'
                  .format(gap[0], gap[1]))
            continue

        lg.info("gap: %s" % repr(gap))
        argv = ['-silent',
                '--wp', "%g" % args.wp,
                '--ws', "%g" % args.ws,
                '--wo', "%g" % args.wo,
                '--wi', "%g" % args.wi,
                '--nomocap',  # added 16/4/2018
                '-v', args.video,
                '--output-n', "%d" % args.output_n]
        if args.w_occlusion:
            argv.extend(['-w-occlusion'])
        if args.no_isec:
            argv.extend(['-no-isec'])

        if args.filter_same_scene:
            argv.extend(['--filter-scenes', name_query.partition('_')[0]])
        # else:
        #     assert False, "crossvalidation assumed"

        if args.maxiter:
            argv.extend(['--maxiter', "%d" % args.maxiter])

        argv.extend(['independent',
                     '-s', args.s,
                     '--gap', "%d" % gap[0], "%d" % gap[1],
                     '--dest-dir', args.dest_dir,
                     '-tc', '-0.1',
                     '--batch-size', "%d" % args.batch_size])
        lg.info("argv: %s" % argv)

        # if 'once' not in locals():
        try:
            _cache_scenes = opt_consistent(argv, cache_scenes)
            if isinstance(_cache_scenes, list) and len(_cache_scenes) \
                and (cache_scenes is None
                     or len(_cache_scenes) != len(cache_scenes)):
                cache_scenes = _cache_scenes
        except FileNotFoundError as e:
            lg.error("e: %s" % e)
            if e.__str__().endswith('_2d_00.json\''):
                from imapper.pose.main_denis import main as opt0
                argv_opt0 = ['s8', '-d', "%s/denis" % d_query,
                             '-smooth', '0.005']
                opt0(argv_opt0)
            else:
                print(e.__str__())

            opt_consistent(argv)

        # once = True
    show_folder([args.video])
    
    extract_gaps([args.video, args.s])

    print("skipped: %s" % skipped)


class DepthTimeCharness(object):
    def __init__(self, pos_3d, frame_id, charness):
        self._frame_id = frame_id
        self._charness = charness
        self._pos_3d = pos_3d
        self._depth = np.linalg.norm(pos_3d)

    @property
    def depth(self):
        return self._depth

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def charness(self):
        return self._charness

    @property
    def pos_3d(self):
        return self._pos_3d


def select_gaps(argv):
    pjoin = os.path.join
    parser = argparse.ArgumentParser("Select gaps")
    parser = parser_video(parser)
    parser.add_argument('--select-gaps', action='store_true')
    parser.add_argument("-s", "--d-scenelets",
                        dest='s', type=argparse_check_exists,
                        help="Folder containing original PiGraphs scenelets")
    args = parser.parse_args(argv)
    d = os.path.join(args.video, args.d)
    assert os.path.exists(d), "does not exist: %s" % d

    if os.path.isdir(args.video):
        if args.video.endswith(os.sep):
            args.video = args.video[:-1]
    name_query = args.video.split(os.sep)[-1]
    assert len(name_query), args.video.split(os.sep)
    p_query = pjoin(args.video, "skel_%s_unannot.json" % name_query) \
        if os.path.isdir(args.video) else args.video
    assert p_query.endswith('.json'), "Need a skeleton file"

    # load initial video path (local poses)
    query = Scenelet.load(p_query, no_obj=True)
    frame_ids = query.skeleton.get_frames()
    centroids = Skeleton.get_resampled_centroids(start=frame_ids[0],
                                                 end=frame_ids[-1],
                                                 old_frame_ids=frame_ids,
                                                 poses=query.skeleton.poses)

    depths_times_charnesses = []
    skeleton = Skeleton()
    depths = []
    for p in sorted(os.listdir(d)):
        ch = float(p.split('charness')[1][1:])
        d_time = pjoin(d, p)
        p_skel = next(f for f in os.listdir(d_time)
                      if os.path.isfile(pjoin(d_time, f))
                      and f.startswith('skel') and f.endswith('json')
                      and '_00' in f)
        sclt = Scenelet.load(pjoin(d_time, p_skel))
        mn, mx = sclt.skeleton.get_frames_min_max()
        frame_id = mn + (mx - mn) // 2
        if query.skeleton.has_pose(frame_id):
            pos_3d = query.skeleton.get_centroid_3d(frame_id)
        else:
            lin_id = frame_id - frame_ids[0]
            pos_3d = centroids[lin_id, :]
        depth = np.linalg.norm(pos_3d)
        depths.append(depth)
        depths_times_charnesses.append(
          DepthTimeCharness(depth=depth, frame_id=frame_id, charness=ch))
    hist, bin_edges = np.histogram(depths, bins=5)
    lg.debug("hist: %s" % hist)
    lg.debug("edges: %s" % bin_edges)


if __name__ == '__main__':
    try:
        if '--select-gaps' in sys.argv:
            select_gaps(sys.argv[1:])
        else:
            main(sys.argv[1:])
    except bdb.BdbQuit as e:
        # exiting from the debugger
        pass
    except SystemExit as e:
        if e.code != 0:
            raise
    except Exception as e:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem()
