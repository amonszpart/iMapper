import os
import sys

from imapper.visualization.plotting import plt
import argparse
import copy
import json
import shutil
from itertools import product
import multiprocessing

import cv2
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
try:
    import shapely.geometry as geom
    from descartes import PolygonPatch
except:
    pass
from builtins import range

import imapper.logic.geometry as gm
from imapper.config.conf import Conf
from imapper.filtering.one_euro_filter import OneEuroFilter
from imapper.input.intrinsics import intrinsics_matrix
from imapper.logic.joints import Joint
from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.pose import config
from imapper.pose.path_labeling import show_images, \
    greedy_actors, more_actors_gurobi, show_multi, DataPosesWrapper
from imapper.pose.opt import optimize_path
from imapper.pose.skeleton import JointDenis
from imapper.util.my_pickle import pickle, pickle_load
from imapper.util.stealth_logging import lg, logging
from imapper.pose.confidence import get_conf_thresholded
from imapper.pose.config import INPUT_SIZE
from imapper.pose.opt import filter_outliers, filter_wrong_poses
from subprocess import check_call, call


def prepare(camera_name, winsorize_limit, shape_orig, path_scene,
            skel_ours_2d, skel_ours, resample, path_skel):
    """

    Args:
        camera_name (str):
            Name of camera for intrinsics calculation.
        winsorize_limit (float):
            Outlier detection threshold.
        shape_orig (Tuple[int, int]):
            Original video resolution.
        path_scene (str): Root path to scene.
        skel_ours_2d (np.ndarray): (N, 2, 16)
            2D skeletons from LFD in our format.
        skel_ours (np.ndarray): (N, 3, 16)
            Local space 3D skeletons in iMapper coordinate frame
            (y-down, z-front).
        resample (bool):
            If needs densification using Blender's IK engine.
    Returns:
        skel_ours (Skeleton):
        skel_ours_2d (Skeleton):
        intrinsics (np.ndarray):
    """
    assert camera_name is not None and isinstance(camera_name, str), \
        "Need a camera name"

    if shape_orig is None:
        shape_orig = (np.float32(1080.), np.float32(1920.))
    np.set_printoptions(linewidth=200, suppress=True)

    if False:
        plt.figure()
        for i, frame_id in enumerate(skel_ours.get_frames()):
            plot_2d(skel_ours_2d.get_pose(frame_id), images[frame_id])
            plt.show()

    path_intrinsics = os.path.join(path_scene, "intrinsics.json")
    if os.path.exists(path_intrinsics):
        lg.warning("Loading existing intrinsics matrix!")
        K = np.array(json.load(open(path_intrinsics, 'r')), dtype=np.float32)
        scale = (
            shape_orig[1]
            / int(round(shape_orig[1] * float(INPUT_SIZE)
                        / shape_orig[0])),
            shape_orig[0] / float(INPUT_SIZE)
        )
        K[0, 0] /= scale[0]
        K[0, 2] /= scale[0]
        K[1, 1] /= scale[1]
        K[1, 2] /= scale[1]
    else:
        K = intrinsics_matrix(INPUT_SIZE, shape_orig, camera_name)
        focal_correction = Conf.get().optimize_path.focal_correction
        if abs(focal_correction - 1.) > 1.e-3:
            lg.warning("Warning, scaling intrinsics matrix by %f"
                       % focal_correction)
            K[0, 0] *= focal_correction
            K[1, 1] *= focal_correction
    #print("K:\n%s,\nintr:\n%s" % (K, intr))
    # sys.exit(0)

    #
    # Prune poses
    #

    skel_ours_2d, frame_ids_removed = filter_outliers(
      skel_ours_2d, winsorize_limit=winsorize_limit, show=False)
    frames_to_remove_3d = filter_wrong_poses(skel_ours_2d, skel_ours)
    frames_to_ignore_list = set()
    # if frames_ignore is not None:
    #     for start_end in frames_ignore:
    #         if isinstance(start_end, tuple):
    #             l_ = list(range(
    #               start_end[0],
    #               min(start_end[1], skel_ours_2d.get_frames()[-1])))
    #             frames_to_remove_3d.extend(l_)
    #             frames_to_ignore_list.update(l_)
    #         else:
    #             assert isinstance(start_end, int), \
    #                 "Not int? %s" % repr(start_end)
    #             frames_to_remove_3d.append(start_end)
    #             frames_to_ignore_list.add(start_end)
    for frame_id in skel_ours.get_frames():
        if frame_id in frames_to_remove_3d:
            skel_ours.remove_pose(frame_id)

    # resample skeleton to fill in missing frames
    skel_ours_old = skel_ours
    frame_ids_filled_in = set(skel_ours_2d.get_frames()).difference(
      set(skel_ours_old.get_frames()))
    if resample:
        lg.warning("Resampling BEFORE optimization")
        # frames_to_resample = sorted(set(skel_ours_2d.get_frames()).difference(
        #   frames_to_ignore_list))
        # skel_ours = Skeleton.resample(skel_ours_old,
        #                               frame_ids=frames_to_resample)
        # Aron on 6/4/2018
        sclt_ours = Scenelet(skeleton=skel_ours)
        stem = os.path.splitext(path_skel)[0]
        path_filtered = "%s_filtered.json" % stem
        path_ipoled = "%s_ikipol.json" % os.path.splitext(path_filtered)[0]
        if not os.path.exists(path_ipoled):
            sclt_ours.save(path_filtered)
            script_filepath = \
                os.path.normpath(os.path.join(
                  os.path.dirname(os.path.abspath(__file__)),
                  os.pardir, 'blender', 'ipol_ik.py'))
            assert os.path.exists(script_filepath), "No: %s" % script_filepath
            blender_path = os.environ.get('BLENDER')
            if not os.path.isfile(blender_path):
                raise RuntimeError(
                    "Need \"BLENDER\" environment variable to be set "
                    "to the blender executable")
            cmd_params = [blender_path, '-noaudio', '-b', '-P',
                          script_filepath, '--', path_filtered]
            print("calling %s" % " ".join(cmd_params))
            ret = check_call(cmd_params)
            print("ret: %s" % ret)
        else:
            lg.warning("\n\n\tNOT recomputing IK interpolation, "
                       "file found at %s!\n" % path_ipoled)
        skel_ours = Scenelet.load(path_ipoled, no_obj=True).skeleton

        # remove extra frames at ends and beginnings of actors
        spans = skel_ours_old.get_actor_empty_frames()
        old_frames = skel_ours_old.get_frames()
        frames_to_remove = []
        for frame_id in skel_ours.get_frames():
            if frame_id not in old_frames:
                in_spans = next((True for span in spans
                                 if span[0] < frame_id < span[1]),
                                None)
                if in_spans:
                    frames_to_remove.append(frame_id)
                    # lg.debug("diff: %s  (a%s, f%s)"
                    #          % (
                    #              frame_id,
                    #              skel_ours_old.get_actor_id(frame_id),
                    #              skel_ours_old.mod_frame_id(frame_id)
                    #          ))
        for frame_id in frames_to_remove:
            skel_ours.remove_pose(frame_id)

    for frame_id in skel_ours_2d.get_frames():
        if not skel_ours.has_pose(frame_id):
            skel_ours_2d.remove_pose(frame_id)
    for frame_id in skel_ours.get_frames():
        if not skel_ours_2d.has_pose(frame_id):
            skel_ours.remove_pose(frame_id)
    frames_set_ours = set(skel_ours.get_frames())
    frames_set_2d = set(skel_ours_2d.get_frames())
    if frames_set_ours != frames_set_2d:
        print("Frame mismatch: %s" % frames_set_ours.difference(frames_set_2d))

    lg.warning("Removing pelvis and neck from 2D input")
    for frame_id in skel_ours_2d.get_frames():
        skel_ours_2d.set_visible(frame_id, Joint.PELV, 0)
        skel_ours_2d.set_visible(frame_id, Joint.NECK, 0)

    return skel_ours, skel_ours_2d, K, frame_ids_filled_in

def load_image(fname):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    shape_orig = image.shape
    scale = config.INPUT_SIZE/(image.shape[0] * 1.0)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    # print("image.size: %s" % repr(image.shape), shape_orig)
    # b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)
    return image, shape_orig


def create_rect(size, angle_deg, pos):
    rect_unit = np.array([[-.5, -.5], [-.5, .5], [.5, .5], [.5, -.5]])
    ang = np.deg2rad(angle_deg)
    cs = np.cos(ang)
    sn = np.sin(ang)
    m0 = \
        np.dot(
           np.array([[1., 0., pos[0]],
                     [0., 1., pos[1]],
                     [0., 0., 1.0]]),
           np.dot(
              np.array([[cs, -sn, 0.],
                        [sn, cs, 0.],
                        [0., 0., 1.]]),
              np.array([[size[0], 0., 0.],
                        [0., size[1], 0.],
                        [0., 0., 1.]])
           )
        )
    return np.dot(m0[:2, :2], rect_unit.T).T + m0[:2, 2]


def filter_(skel_opt, out_filter_path=None, skel_orig=None, weight_smooth=None,
            forwards_window_size=0):
    # was_writeable = skel_opt.poses.flags.writeable
    # skel_opt.poses.flags.writeable = False
    if not os.path.exists(out_filter_path):
        os.mkdir(out_filter_path)
    # else:
    #     shutil.rmtree(out_filter_path)
    #     os.mkdir(out_filter_path)
    # mincutoffs = itertools.chain(
    #     np.linspace(0.001, 0.101, num=21, endpoint=True),
    #     np.linspace(0.1, 1.1, num=11, endpoint=True))
    # betas = itertools.chain(
    #     np.linspace(0.001, 0.101, num=21, endpoint=True),
    #     np.linspace(0.1, 1.1, num=11, endpoint=True))

    skel_out = copy.deepcopy(skel_opt)
    skel_final_out = None
    # 1-euro filter low-pass threshold parameter
    mincutoffs = [0.025]  # [0.05, 0.2, 0.3]
    betas = [.5]  # [0.25, 0.5, 0.75, 1., 1.5, 2.]
    # forward estimation window size (+-k, so k=1 means a window of 3)
    ks = [forwards_window_size]  # {0, 1, 2}
    for mincutoff, (k, beta) in product(mincutoffs, product(ks, betas)):
        config = Conf.get().one_euro.__dict__
        config['mincutoff'] = mincutoff
        config['beta'] = beta
        oefs = [OneEuroFilter(**config), OneEuroFilter(**config)]

        frames = skel_opt.get_frames()
        filtered = np.array(
            [
                [
                    oefs[0](p[0], timestamp=frames[i]),
                    p[1],
                    oefs[1](p[2], timestamp=frames[i])
                ]
                for i, p in enumerate(skel_opt.poses[:, :, Joint.PELV])
            ]
        )

        # estimate translation
        t = filtered - skel_opt.poses[:, :, Joint.PELV]
        assert t.shape[1] == 3, "Wrong shape: %s" % repr(t.shape)
        # apply translation
        for i, frame_id in enumerate(skel_opt.get_frames()):
            transform = gm.translation(t[i])
            skel_out.set_pose(
                frame_id=frame_id,
                pose=gm.htransform(
                    transform, skel_opt.get_pose(frame_id=frame_id))
            )

        skel_out.estimate_forwards(k=k)

        #
        # Plot
        #
        if out_filter_path is not None:
            tmp = copy.deepcopy(skel_opt)
            tmp.estimate_forwards(k=k)

            if show:
                fig = plt.figure(figsize=(24, 12))
                ax0 = plt.subplot2grid((1, 2), (0, 0), aspect='equal')
                ax1 = plt.subplot2grid((1, 2), (0, 1), aspect='equal',
                                       sharey=ax0, sharex=ax0)
                legends = {ax0: [], ax1: []}
                if skel_orig is not None:
                    h = ax0.plot(skel_orig.poses[:, 0, Joint.PELV],
                                 skel_orig.poses[:, 2, Joint.PELV], 'ks',
                                 markerfacecolor='none',
                                 label='Original')[0]
                    legends[ax0].append(h)
                h = ax0.plot(tmp.poses[:, 0, Joint.PELV],
                             tmp.poses[:, 2, Joint.PELV], 'b--x',
                             label='Resampled')[0]
                legends[ax0].append(h)
                ax0.legend(handles=legends[ax0])
                ax0.set_title(
                    "Resampled (smoothness weight: %g, forward window size: %d)"
                    % (args.smooth, k*2+1))

                legends[ax1].append(
                    ax1.plot(tmp.poses[:, 0, Joint.PELV],
                             tmp.poses[:, 2, Joint.PELV], 'b--x',
                             label='Resampled')[0])
                legends[ax1].append(
                    ax1.plot(skel_out.poses[:, 0, Joint.PELV],
                             skel_out.poses[:, 2, Joint.PELV],
                             'k--x', label='Filtered')[0])
                ax1.set_title("Filtered (fc_min: %g, beta: %g)"
                              % (config['mincutoff'], config['beta']))

                ax1.legend(handles=legends[ax1])

                for frame_id in skel_out.get_frames():
                    pos = tmp.get_joint_3d(joint_id=Joint.PELV, frame_id=frame_id)
                    fw = tmp.get_forward(frame_id=frame_id, estimate_ok=False) \
                         * 0.1
                    ax0.arrow(pos[0], pos[2], fw[0], fw[2], fc='y', ec='y',
                              head_width=0.02)

                    pos = skel_out.get_joint_3d(joint_id=Joint.PELV, frame_id=frame_id)
                    fw = skel_out.get_forward(frame_id=frame_id, estimate_ok=False) * 0.1
                    ax1.arrow(pos[0], pos[2], fw[0], fw[2], fc='g', ec='g',
                              head_width=0.02)
                ax0.set_xlim(-1.5, 2.)
                ax1.set_xlim(-1.5, 2.)
                ax0.set_ylim(1.75, 5.75)
                ax1.set_ylim(1.75, 5.75)

                # plt.legend(handles=legend)
                # plt.title("Mincutoff: %g" % mincutoff)
                # plt.title("Mincutoff: %g, beta: %g"
                #           % (config['mincutoff'], config['beta']))
                if weight_smooth is not None:
                    name_file = "_wsmooth_%.3f" % weight_smooth
                else:
                    name_file = ""
                name_file = "filtered%s_k_%d_mco_%.3f_beta_%.3f.png" \
                            % (name_file, k, config['mincutoff'], config['beta'])

                plt.savefig(os.path.join( out_filter_path, name_file))
                plt.close()

        # save first run as output
        if skel_final_out is None:
            skel_final_out = skel_out

        # prepare for next
        skel_out = copy.deepcopy(skel_opt)

    # skel_opt.poses.flags.writeable = was_writeable
    return skel_final_out


def _load_chunk(path_file):
    lg.debug("Opening %s" % path_file)
    with open(path_file, 'rb') as fil:
        return pickle.load(fil)


def load_images(path_images, n=50):
    images = {}
    shape_orig = None
    logging.info("Reading images...")
    path_images_pickle = os.path.join(path_images, 'images_0.pickle')
    if os.path.exists(path_images_pickle):
        logging.warning("Reading PICKLE from %s" % path_images_pickle)
        _gen = (os.path.join(path_images, path_file)
                for path_file in os.listdir(path_images)
                if path_file.endswith('pickle')
                and path_file.startswith('images_'))

        pool = multiprocessing.Pool(4)
        processes = []
        for path_file in _gen:
            processes.append(pool.apply_async(
               func=_load_chunk, args=[path_file]))
        pool.close()
        pool.join()
        for process in processes:
            pickled = process.get()
            if isinstance(pickled, tuple) and len(pickled) == 2:
                images.update(pickled[0])
                shape_orig = pickled[1]
            else:
                images.update(pickled)
        assert shape_orig is not None, \
            "Shape orig must be in one of the files"
    else:
        # load images
        _gen = (f for f in os.listdir(path_images)
                if f.endswith('jpg'))
        for f in _gen:
            frame_id = int(os.path.splitext(f)[0].split('_')[1])
            images[frame_id], shape_orig = \
                load_image(os.path.join(path_images, f))
        del frame_id, f, _gen
        #
        # save pickles
        #

        # sorted list of frame_ids
        keys = sorted(list(images.keys()))
        for id_pickle, i in enumerate(range(0, len(keys), n)):
            range_end = min(i+n, len(keys))
            _keys = keys[i:range_end]
            _images = {frame_id: images[frame_id]
                       for frame_id in _keys}
            path_file = os.path.join(path_images,
                                     "images_%d.pickle" % id_pickle)
            with open(path_file, 'wb') as file_out:
                to_dump = (_images, shape_orig) \
                    if not i \
                    else _images
                pickle.dump(to_dump, file_out, protocol=-1)
                lg.debug("Saved to %s" % path_file)

        # pickle.dump((images, shape_orig), open(path_images_pickle, 'wb'))
        # logging.info("Saved PICKLE to %s" % path_images_pickle)
    logging.info("Finished loading images...")

    return images, shape_orig

def cleanup(data, p_dir):
    """Removes small skeletons (<=3 confident joints).

    Returns:
        data (dict):
            Denis poses without the pose_ids that have been deleted.
        manual (dict):
            Information to actor segmenter, which ones to assign to a specific
            actor.
        first_run (bool):
            Whether the pose deletion actually happened. If True, all poses
            are saved, so that we can inspect the pose_ids in color in
            "debug/labeling_orig".
    """
    keys = ('visible', 'pose_3d', 'visible_float', 'centered_3d', 'pose_2d')
    p_to_remove = os.path.join(p_dir, 'pose_ids.json')
    # to_remove = {1: [1], 6: [0], 8: [1], 9: [1], 19: [1], 20: [1], 21: [1],
    # 22: [1], 23: [1], 24: [1], 25: [1]}
    if not os.path.exists(p_to_remove):
        manual = {'delete': {}, 'labels': {}}
    else:
        manual = json.load(open(p_to_remove, 'r'))
    to_remove = manual['delete']
    # to_remove = {}

    for frame_str in sorted(data):
        try:
            frame_id = int(frame_str.split('_')[1])
        except ValueError:
            print("skipping key %s" % frame_id)
            continue
        if frame_str not in to_remove:
            to_remove[frame_str] = []
        pose_in_2d = np.array(data[frame_str][u'pose_2d'])
        vis_f = np.array(data[frame_str][u'visible_float'])
        for pose_id in range(pose_in_2d.shape[0]):
            conf = get_conf_thresholded(vis_f[pose_id, ...], None, np.float32)
            cnt = np.sum(conf > 0.5)
            if cnt <= 3:
                try:
                    if pose_id not in to_remove[frame_str]:
                        to_remove[frame_str].append(pose_id)
                except KeyError:
                    to_remove[frame_str] = [pose_id]
    p_to_remove_out = os.path.join(p_dir, 'pose_ids.auto.json')
    first_run = not os.path.exists(p_to_remove_out)
    json.dump({'delete': to_remove, 'labels': {}},
               open(p_to_remove_out, 'w'), indent=4)

    if not first_run:
        for frame_str, pose_ids in to_remove.items():
            # frame_str = "color_%05d" % int(frame_id)
            if len(pose_ids) == len(data[frame_str]['visible']):
                del data[frame_str]
                continue
            frame_id = int(frame_str.split('_')[1])
            assert frame_str in data, "no"
            for key in keys:
                tmp = data[frame_str][key]
                assert isinstance(tmp, list), "wrong type"
                data[frame_str][key] = list()
                for lin_id, entry in enumerate(tmp):
                    if lin_id not in pose_ids:
                        data[frame_str][key].append(entry)

    return data, manual, first_run


def main(argv):
    conf = Conf.get()
    parser = argparse.ArgumentParser("Denis pose converter")
    parser.add_argument('camera_name', help="Camera name ('G15', 'S6')",
                        type=str)
    parser.add_argument(
        '-d', dest='dir', required=True,
        help="Path to the <scene folder>/denis containing skeletons.json")
    parser.add_argument(
        '-filter', dest='with_filtering', action="store_true",
        help="Should we do post-filtering (1-euro) on the pelvis positions")
    parser.add_argument(
        '-huber', required=False, help="Should we do huber loss?",
      action='store_true')
    parser.add_argument(
        '-smooth', type=float, default=0.005,
        help="Should we have a smoothness term (l2/huber)?")
    parser.add_argument(
        '--winsorize-limit', type=float,
        default=conf.optimize_path.winsorize_limit,
        help='Threshold for filtering too large jumps of the 2D centroid'
    )
    parser.add_argument(
      '--no-resample', action='store_true',
      help="add resampled frames"
    )
    parser.add_argument(
      '--n-actors', type=int, default=1, help="How many skeletons to track."
    )
    parser.add_argument('-n-actors', type=int, default=1,
                        help="Max number of people in scene.")
    # parser.add_argument(
    #     '-r', type=float,
    #     help='Video rate. Default: 1, if avconv -r 5. '
    #          'Original video sampling rate (no subsampling) should be '
    #          '24/5=4.8. avconv -r 10 leads to 24/10=2.4.',
    #     required=True)
    parser.add_argument(
        '--person_height', type=float,
        help='Assumed height of human(s) in video.',
        default=Conf.get().optimize_path.person_height
    )
    parser.add_argument(
        '--forwards-window-size', type=int,
        help='How many poses in time to look before AND after to '
             'average forward direction. 0 means no averaging. Default: 0.',
        default=0
    )
    parser.add_argument(
        '--no-img', action='store_true',
        help='Read and write images (vis reproj error)')
    parser.add_argument('--postfix', type=str, help="output file postfix.",
                        default='unannot')
    args = parser.parse_args(argv)
    show = False
    args.resample = not args.no_resample
    # assert not args.resample, "resample should be off"
    assert os.path.exists(args.dir), "Source does not exist: %s" % args.dir
    p_scene = os.path.normpath(os.path.join(args.dir, os.pardir))  # type: str
    p_video_params = os.path.join(p_scene, 'video_params.json')
    assert os.path.exists(p_video_params), "Need video_params.json for rate"
    if 'r' not in args or args.r is None:
        args.r = json.load(open(p_video_params, 'r'))['rate-avconv']

    # manual parameters (depth initialization, number of actors)
    p_scene_params = os.path.join(args.dir, os.pardir, 'scene_params.json')
    if not os.path.exists(p_scene_params):
        scene_params = {'depth_init': 10., 'actors': args.n_actors,
                        'ground_rot': [0., 0., 0.]}
        json.dump(scene_params, open(p_scene_params, 'w'))
        raise RuntimeError("Inited scene_params.json, please check: %s"
                           % p_scene_params)
    else:
        scene_params = json.load(open(p_scene_params, 'r'))
        lg.warning("Will work with %d actors and init depth to %g"
                   % (scene_params['actors'], scene_params['depth_init']))
        assert '--n-actors' not in argv \
               or args.n_actors == scene_params['actors'], \
            "Actor count mismatch, remove %d from args, because " \
            "scene_params.json says %d?" \
            % (args.n_actors, scene_params['actors'])
        args.n_actors = scene_params['actors']
        ground_rot = scene_params['ground_rot'] or [0., 0., 0.]

    # load images
    path_images = os.path.abspath(os.path.join(args.dir, os.pardir, 'origjpg'))
    images = {}
    shape_orig = None
    if not args.no_img:
        images, shape_orig = load_images(path_images)

    path_skeleton = \
        max((f for f in os.listdir(os.path.join(args.dir))
             if f.startswith('skeletons') and f.endswith('json')),
            key=lambda s: int(os.path.splitext(s)[0].split('_')[1]))
    print("path_skeleton: %s" % path_skeleton)
    data = json.load(open(os.path.join(args.dir, path_skeleton), 'r'))
    # data, pose_constraints, first_run = \
    #     cleanup(data, p_dir=os.path.join(args.dir, os.pardir))
    # poses_2d = []
    # plt.figure()
    # show_images(images, data)
    if False:
        # pose_ids = identify_actors_multi(data, n_actors=1)
        p_segm_pickle = os.path.join(args.dir, os.pardir,
                                     "label_skeletons.pickle")
        problem = None
        if False and os.path.exists(p_segm_pickle):
            lg.warning("Loading skeleton segmentation from pickle %s"
                       % p_segm_pickle)
            pose_ids, problem = pickle_load(open(p_segm_pickle, 'rb'))
        if not problem or problem._n_actors != args.n_actors:
            pose_ids, problem, data = more_actors_gurobi(
              data, n_actors=args.n_actors, constraints=pose_constraints,
              first_run=first_run)
            if True or show:
                show_multi(images, data, pose_ids, problem,
                           p_dir=os.path.join(args.dir, os.pardir),
                           first_run=first_run, n_actors=args.n_actors)
            pickle.dump((pose_ids, problem), open(p_segm_pickle, 'wb'), -1)
    else:
        pose_ids = greedy_actors(data, n_actors=args.n_actors)
        data = DataPosesWrapper(data=data)

    visible_f = {a: {} for a in range(args.n_actors)}
    visible_f_max = 0.
    if show:
        plt.ion()
        fig = None
        axe = None
        scatters = dict()

    # how many images we have
    min_frame_id = min(f for f in pose_ids)
    frames_mod = max(f for f in pose_ids) - min_frame_id + 1
    skel_ours = Skeleton(frames_mod=frames_mod, n_actors=args.n_actors,
                         min_frame_id=min_frame_id)
    skel_ours_2d = Skeleton(frames_mod=frames_mod, n_actors=args.n_actors,
                            min_frame_id=min_frame_id)

    # assert len(images) == 0 or max(f for f in images) + 1 == frames_mod, \
    #     "Assumed image count is %d, but max_frame_id is %d" \
    #     % (len(images), frames_mod-1)
    if isinstance(data, DataPosesWrapper):
         frames = data.get_frames()
    else:
        frames = []
        for frame_str in sorted(data.get_frames()):
            try:
                frame_id = int(frame_str.split('_')[1])
            except ValueError:
                print("skipping key %s" % frame_id)
                continue
            frames.append(frame_id)
    my_visibilities = [[], []]
    for frame_id in frames:
        frame_str = DataPosesWrapper._to_frame_str(frame_id)
        pose_in = data.get_poses_3d(frame_id=frame_id)
        # np.asarray(data[frame_str][u'centered_3d'])
        # pose_in_2d = np.asarray(data[frame_str][u'pose_2d'])
        pose_in_2d = data.get_poses_2d(frame_id=frame_id)
        # visible = np.asarray(data[frame_str][u'visible'])

        if False and len(pose_in.shape) > 2:
            pose_id = pose_ids[frame_id]
            if not args.no_img:
                im = cv2.cvtColor(images[frame_id], cv2.COLOR_RGB2BGR)
                for i in range(pose_in.shape[0]):
                    c = (1., 0., 0., 1.)
                    if i == pose_id:
                        c = (0., 1., 0., 1.)
                    color = tuple(int(c_ * 255) for c_ in c[:3])
                    for p2d in pose_in_2d[i, :, :]:
                        # color = (c[0] * 255, c[1] * 255., c[2] * 255.)
                        cv2.circle(
                            im, (p2d[1], p2d[0]), radius=3, color=color, thickness=-1)
                    center = np.mean(pose_in_2d[i, :, :], axis=0).round().astype('i4').tolist()
                    cv2.putText(im, "%d" % i, (center[1], center[0]), 1, 1, color)
                if show:
                    cv2.imshow("im", im)
                    cv2.waitKey(100)
            # if sid not in scatters:
            #     scatters[sid] = axe.scatter(pose_in_2d[i, :, 1], pose_in_2d[i, :, 0], c=c)
            # else:
            #     scatters[sid].set_offsets(pose_in_2d[i, :, [1, 0]])
            #     scatters[sid].set_array(np.tile(np.array(c), pose_in_2d.shape[1]))
                # scatter.set_color(c)
            # plt.draw()
            # plt.pause(1.)
            pose_in = pose_in[pose_id, :, :]
            pose_in_2d = pose_in_2d[pose_id, :, :]
            visible = visible[pose_id]
        # else:
            # pose_id = 0
            # pose_id = pose_ids[frame_id]

        for actor_id in range(args.n_actors):
            # if actor_id in (2, 3, 4, 5, 8, 9)
            # expanded frame_id
            frame_id2 = Skeleton.unmod_frame_id(frame_id=frame_id,
                                                actor_id=actor_id,
                                                frames_mod=frames_mod)
            assert (actor_id != 0) ^ (frame_id2 == frame_id), "no"
            frame_id_mod = skel_ours.mod_frame_id(frame_id=frame_id2)

            assert frame_id_mod == frame_id, \
                "No: %d %d %d" % (frame_id, frame_id2, frame_id_mod)
            actor_id2 = skel_ours.get_actor_id(frame_id2)
            assert actor_id2 == actor_id, "no: %s %s" % (actor_id, actor_id2)

            # which pose explains this actor in this frame
            pose_id = pose_ids[frame_id][actor_id]
            # check, if actor found
            if pose_id < 0:
                continue

            # 3D pose
            pose = pose_in[pose_id, :, JointDenis.revmap].T
            # added by Aron on 4/4/2018 (Denis' pelvis is too high up)
            pose[:, Joint.PELV] = (pose[:, Joint.LHIP] + pose[:, Joint.RHIP]) \
                                  / 2.
            skel_ours.set_pose(frame_id2, pose)

            # 2D pose
            pose_2d = pose_in_2d[pose_id, :, :]
            arr = np.array(JointDenis.pose_2d_to_ours(pose_2d),
                           dtype=np.float32).T
            skel_ours_2d.set_pose(frame_id2, arr)

            #
            # visibility (binary) and confidence (float)
            #

            # np.asarray(data[frame_str][u'visible'][pose_id])
            vis_i = data.get_visibilities(frame_id)[pose_id]

            # vis_f = np.asarray(data[frame_str][u'visible_float'][pose_id])
            vis_f = data.get_confidences(frame_id)[pose_id]
            for jid, visible in enumerate(vis_i): # for each joint
                # binary visibility
                jid_ours = JointDenis.to_ours_2d(jid)
                skel_ours_2d.set_visible(frame_id2, jid_ours, visible)

                # confidence (fractional visibility)
                if np.isnan(vis_f[jid]):
                    continue

                try:
                    visible_f[actor_id][frame_id2][jid_ours] = vis_f[jid]
                except KeyError:
                    visible_f[actor_id][frame_id2] = {jid_ours: vis_f[jid]}
                visible_f_max = max(visible_f_max, vis_f[jid])
                conf_ = get_conf_thresholded(vis_f[jid],
                                             thresh_log_conf=None,
                                             dtype_np=np.float32)
                skel_ours_2d.set_confidence(frame_id=frame_id2, joint=jid_ours,
                                            confidence=conf_)
                my_visibilities[0].append(vis_f[jid])
                my_visibilities[1].append(conf_)
            skel_ours_2d._confidence_normalized = True

    plt.figure()
    plt.plot(my_visibilities[0], my_visibilities[1], 'o')
    plt.savefig('confidences.pdf')
    
    assert skel_ours.n_actors == args.n_actors, "no"
    assert skel_ours_2d.n_actors == args.n_actors, "no"
    # align to room
    min_z = np.min(skel_ours.poses[:, 2, :])
    print("min_max: %s, %s" %
          (min_z, np.max(skel_ours.poses[:, 2, :])))
    skel_ours.poses[:, 2, :] += min_z
    skel_ours.poses /= 1000.
    # The output is scaled to 2m by Denis.
    # We change this to 1.8 * a scale in order to correct for
    # the skeletons being a bit too high still.
    skel_ours.poses *= \
        args.person_height * conf.optimize_path.height_correction / 2.
    skel_ours.poses[:, 2, :] *= -1.
    skel_ours.poses = skel_ours.poses[:, [0, 2, 1], :]

    # refine
    name_video = args.dir.split(os.sep)[-2]
    out_path = os.path.join(args.dir, os.pardir,
                            "skel_%s_%s.json" % (name_video, args.postfix))
    out_path_orig = os.path.join(args.dir, os.pardir,
                            "skel_%s_lfd_orig.json" % name_video)
    sclt_orig = Scenelet(skeleton=copy.deepcopy(skel_ours))
    sclt_orig.save(out_path_orig)

    skel_ours_2d_all = copy.deepcopy(skel_ours_2d)
    assert len(skel_ours_2d_all.get_frames()), skel_ours_2d_all.get_frames()

    #
    # Optimize
    #

    # frames_ignore = [(282, 372), (516, 1000)]

    skel_ours, skel_ours_2d, intrinsics, \
    frame_ids_filled_in = prepare(
      args.camera_name,
      winsorize_limit=args.winsorize_limit,
      shape_orig=shape_orig,
      path_scene=p_scene,
      skel_ours_2d=skel_ours_2d,
      skel_ours=skel_ours,
      resample=args.resample,
    path_skel=path_skeleton)
    frames_ignore = []
    tr_ground = np.eye(4, dtype=np.float32)
    skel_opt, out_images, K = \
        optimize_path(
          skel_ours, skel_ours_2d, images, intrinsics=intrinsics,
          path_skel=out_path, shape_orig=shape_orig,
          use_huber=args.huber, weight_smooth=args.smooth,
          frames_ignore=frames_ignore, resample=args.resample,
          depth_init=scene_params['depth_init'],
          ground_rot=ground_rot)

    for frame_id in skel_opt.get_frames():
        skel_opt.set_time(frame_id=frame_id, time=float(frame_id) / args.r)

    skel_opt_raw = copy.deepcopy(skel_opt)
    skel_opt_resampled = Skeleton.resample(skel_opt)

    # Filter pelvis
    if args.with_filtering:
        out_filter_path = os.path.join(args.dir, os.pardir, "vis_filtering")
        skel_opt = filter_(skel_opt_resampled, out_filter_path=out_filter_path,
                           skel_orig=skel_opt, weight_smooth=args.smooth,
                           forwards_window_size=args.forwards_window_size)
    else:
        skel_opt.estimate_forwards(k=args.forwards_window_size)
        skel_opt_resampled.estimate_forwards(k=args.forwards_window_size)


    # if len(images):
    #     skel_opt.fill_with_closest(images.keys()[0], images.keys()[-1])

    min_y, max_y = skel_opt.get_min_y(tr_ground)
    print("min_y: %s, max_y: %s" % (min_y, max_y))

    #
    # save
    #
    frame_ids_old = set(skel_opt.get_frames())
    if args.resample:
        skel_opt = skel_opt_resampled
        frame_ids_filled_in.update(set(skel_opt.get_frames()).difference(
          frame_ids_old))
        lg.warning("Saving resampled scenelet!")
    scenelet = Scenelet(skel_opt)
    del skel_opt
    # skel_dict = skel_opt.to_json()
    tr_ground[1, 3] = min_y
    scenelet.aux_info['ground'] = tr_ground.tolist()
    assert isinstance(ground_rot, list) and len(ground_rot) == 3
    scenelet.add_aux_info('ground_rot', ground_rot)
    scenelet.add_aux_info(
       'path_opt_params', {
           'rate': args.r,
           'w-smooth': args.smooth,
           'winsorize-limit': args.winsorize_limit,
           'camera': args.camera_name,
           'huber': args.huber,
           'height_correction': conf.optimize_path.height_correction,
           'focal_correction': conf.optimize_path.focal_correction
       }
    )
    scenelet.add_aux_info('frame_ids_filled_in',
                          list(frame_ids_filled_in))

    # To MATLAB
    # _skeleton.get_min_y(_tr_ground)
    # with skel_opt as skeleton:
    # skeleton = skel_opt
    # skeleton_name = os.path.split(args.dir)[0]
    # skeleton_name = skeleton_name[skeleton_name.rfind('/')+1:]
    # mdict = skeleton.to_mdict(skeleton_name)
    # mdict['room_transform'] = tr_ground
    # mdict['room_transform'][1, 3] *= -1.
    # print(mdict)
    # print("scene_name?: %s" % os.path.split(args.dir)[0])
    # skeleton.save_matlab(
    #     os.path.join(os.path.dirname(args.dir), "skeleton_opt.mat"),
    #     mdict=mdict)

    assert scenelet.skeleton.has_forwards(), "No forwards??"
    scenelet.save(out_path)
    if show:
        # save path plot
        out_path_path = os.path.join(
            args.dir, os.pardir, "%s_path.jpg" % name_video)
        path_fig = plot_path(scenelet.skeleton)
        legend = ["smooth %g" % args.smooth]

    # hack debug
    # path_skel2 = os.path.join(args.dir, os.pardir, 'skel_lobby7_nosmooth.json')
    # if os.path.exists(path_skel2):
    #     skel2 = Skeleton.load(path_skel2)
    #     path_fig = plot_path(skel2, path_fig)
    #     legend.append('no smooth')
    if show:
        plt.legend(legend)
        path_fig.savefig(out_path_path)

    # backup args
    path_args = os.path.join(args.dir, os.pardir, 'args_denis.txt')
    with open(path_args, 'a') as f_args:
        f_args.write("%s %s\n" % (os.path.basename(sys.executable),
                                  " ".join(argv)))
        
    # save 2D detections to file
    if args.postfix == 'unannot':
        path_skel_ours_2d = os.path.join(
            args.dir, os.pardir,
            "skel_%s_2d_%02d.json" % (name_video, 0))
        sclt_2d = Scenelet(skel_ours_2d_all)
        print('Saving {} to {}'.format(len(skel_ours_2d_all.get_frames()),
                                       path_skel_ours_2d))
        sclt_2d.skeleton.aux_info = {}
        sclt_2d.save(path_skel_ours_2d)
    else:
        print(args.postfix)

    logging.info("Saving images...")
    if len(images) and len(out_images):
        path_out_images = os.path.join(args.dir, os.pardir, 'color')
        try:
            os.makedirs(path_out_images)
        except OSError:
            pass
        visible_f_max_log = np.log(visible_f_max)
        frames = list(out_images.keys())
        for frame_id in range(frames[0], frames[-1]+1):
            im = out_images[frame_id] if frame_id in out_images \
                else cv2.cvtColor(images[frame_id], cv2.COLOR_BGR2RGB)
            for actor_id in range(args.n_actors):
                if frame_id in visible_f[actor_id]:
                    frame_id2 = skel_ours_2d_all.unmod_frame_id(
                      frame_id=frame_id, actor_id=actor_id,
                      frames_mod=skel_ours_2d_all.frames_mod)
                    for joint, is_vis in visible_f[actor_id][frame_id].items():
                        p2d = skel_ours_2d_all.get_joint_3d(joint,
                                                            frame_id=frame_id2)
                        # radius = np.log(is_vis) / visible_f_max_log
                        # lg.debug("r0: %g" % radius)
                        # radius = np.exp(np.log(is_vis) / visible_f_max_log)
                        # lg.debug("radius is %g" % radius)
                        vis_bool = True
                        if skel_ours_2d_all.has_visible(frame_id=frame_id2,
                                                        joint_id=joint):
                            vis_bool &= skel_ours_2d_all.is_visible(
                              frame_id2, joint)
                        radius = abs(np.log(is_vis / 0.1 + 1e-6))
                        if not np.isnan(radius):
                            p2d = (int(round(p2d[0])), int(round(p2d[1])))
                            cv2.circle(im, center=p2d,
                                       radius=int(round(radius)),
                                       color=(1., 1., 1., 0.5), thickness=1)
                            conf = get_conf_thresholded(conf=is_vis,
                                                        thresh_log_conf=None,
                                                        dtype_np=np.float32)
                            if conf > 0.5:
                                cv2.putText(
                                    img=im,
                                    text=Joint(joint).get_name(),
                                    org=p2d, fontFace=1, fontScale=1,
                                    color=(10., 150., 10., 100.)
                            )
                    # lg.debug("set visibility to %g, radius %g" % (is_vis, radius))
            # if frame_id in out_images:
            scale = (shape_orig[1] / float(im.shape[1]),
                     shape_orig[0] / float(im.shape[0]))
            cv2.imwrite(os.path.join(path_out_images, "color_%05d.jpg"
                                     % frame_id),
                        cv2.resize(im, (0, 0),
                                   fx=scale[0], fy=scale[1],
                                   interpolation=cv2.INTER_CUBIC))
            # else:
            #     fname = "color_%05d.jpg" % frame_id
            #     shutil.copyfile(
            #         os.path.join(path_images, fname),
            #         os.path.join(path_out_images, fname))
        lg.info("Wrote images to %s/" % path_out_images)

        # if 'scale' in locals():
        #     K_scaled = K.copy()
        #     K_scaled[0, 0] *= scale[0]
        #     K_scaled[0, 2] *= scale[0]
        #     K_scaled[1, 1] *= scale[1]
        #     K_scaled[1, 2] *= scale[1]
        #     path_intrinsics = os.path.join(args.dir, os.pardir, "intrinsics.json")
        #     with open(path_intrinsics, 'w') as f_intrinsics:
        #         json.dump(K_scaled.tolist(), f_intrinsics)


    # cmd = "export PATH=/usr/bin:/usr/sbin:/bin:/sbin:$PATH && " \
    #       "/bin/bash -c \"cd /mnt/data_ssd/workspace/stealth/actionscene/ && " \
    #       "~/matlab_symlinks/matlab -r \\\"name_scene='%s'; " \
    #       "compute_pigraph_and_video_scenelets; exit\\\" -nodesktop\"" \
    #       % name_video
    # print(cmd)
    # os.system(cmd)


if __name__ == '__main__':
    main(sys.argv[1:])

# /data/code/openpose# build/examples/openpose/openpose.bin --image_dir /media/data/amonszpa/stealth/shared/video_recordings/angrymen00/origjpg/ --write_json output/ --display 0 --write_images output
