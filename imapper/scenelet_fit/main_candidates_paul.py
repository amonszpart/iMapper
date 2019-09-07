import argparse
import copy
import multiprocessing
import os
import shutil
import sys
from itertools import product
from typing import Dict, List

from imapper.visualization.plotting import matplotlib, plt

import numpy as np
import scipy.io
import scipy.signal
import shapely.geometry as geom
from descartes import PolygonPatch

import imapper.logic.geometry as gm
from imapper.logic.joints import Joint
from imapper.logic.mesh_OBJ import MeshOBJ
from imapper.logic.scenelet import Scenelet, get_scenelets, read_scenelets
from imapper.logic.skeleton import Skeleton
from imapper.scenelet_fit.radial_histograms import SquareHistogram
from imapper.spatial.charness import parse_charness_histograms
from imapper.util.my_pickle import pickle, pickle_load, hash_path_md5
from imapper.util.stealth_logging import lg, split_path
from imapper.logic.categories import CATEGORIES
from scipy.sparse import coo_matrix

try:
    from imapper.visualization.vis_skeleton import VisSkeleton
    from imapper.visualizer.visualizer import Visualizer
except ImportError as e:
    print("Vtk error %s" % e)
from matplotlib.colors import Normalize as cmNormalize
import matplotlib.cm as cm
import matplotlib.patches as mpatches


class NotEnoughMatchesError(RuntimeError):
    def __init__(self, *args):
        super(NotEnoughMatchesError, self).__init__(*args)
        lg.error("[NotEnoughMatchesError] %s" % repr(args))


def vis_input(py_recordings, labels=frozenset({'couch', 'chair', 'table'})):
    vis = Visualizer()
    vis.add_coords()
    seen_scenes = []
    for name_recording, name_scenelet, scenelet in get_scenelets(py_recordings):
        name_scene = Scenelet.get_scene_name_from_recording_name(name_recording)
        if name_scene in seen_scenes:
            continue
        else:
            seen_scenes.append(name_scene)
        # if name_scene != 'ahouse':
        #     continue
        # if not ('ahouse' in name_recording
        #         or 'living3' in name_recording
        #         or 'gates381' in name_recording):
        #     continue
        vis.remove_all_actors(prefixes={"part_"})
        for obj_id, obj in scenelet.objects.items():
            for part_id, part in obj.parts.items():
                vis.add_mesh(MeshOBJ.from_obb(part.obb),
                             "part_%s_%s" % (obj_id, part_id))

        for obj_id, obj in \
            (e for e in scenelet.objects.items() if str(e[1].label) in labels):
            part_id, part = obj._largest_part(with_part_id=True)
            vis.add_arrows(np.squeeze(part.obb.centroid),
                           obj.get_transform(), "ob_coord_%s" % obj_id)
            # mdict.append({'scene': recording,
            #               'oid': obj_id, 'label': obj.label, 'obb': part.obb})
            lg.info(
                "Showing %s %s ob_id: %d, part_id: %d, part.label: %s, label: %s\ntransform:%s"
                % (name_recording, name_scenelet, obj_id, part_id, part.label, obj.label, part.obb._axes))
            vis.show()
            vis.remove_all_actors(prefixes={"ob_coord"})


def fix_leveling(py_scenes, path):
    to_remove = []
    for name_scene in py_scenes:
        print("name scene: %s" % name_scene)
        parts = name_scene.split('__')
        if not parts[1].startswith('skel'):
            parts[1] = "skel_%s" % parts[1]
        print("parts: %s" % parts)
        path_new = os.path.join(path, parts[0])
        try:
            os.makedirs(path_new)
        except OSError:
            pass
        for name_sclt, sclt in py_scenes[name_scene].items():
            sclt.save(os.path.join(path_new, parts[1]))
        to_remove.append(name_scene)
    print(path)
    for name_recording in to_remove:
        shutil.rmtree(os.path.join(path, name_recording))


def read_distances(path):
    assert os.path.exists(path), \
        "Does not exist: %s" % path
    dmat = scipy.io.loadmat(path)
    # lg.info("distances keys: %s" % dmat.keys())
    # print(dmat['pigraph_histogram_charness'])
    # print("shape: %s" % repr(dmat['pigraph_histogram_charness'].shape))
    return dmat['dists'].astype('f4'), \
        [str(n[0][0]) for n in dmat['pigraph_scenelet_names']], \
        [str(n[0][0]) for n in dmat['video_scenelet_names']]


def read_charness(path, return_hists=False, return_names=False):
    assert not (return_hists and return_names), \
        "Can't do both because of pickle unpack order."

    hash_curr = hash_path_md5(path)
    path_pickle = "%s.pickle" % path
    success = False
    if os.path.exists(path_pickle):
        lg.warning("Loading charness from %s" % path_pickle)
        tmp = pickle_load(open(path_pickle, 'rb'))
        if tmp[-1] != hash_curr:
            lg.warning("Hashes don't match (%d, %d), reloading %s..."
                       % (tmp[-1], hash_curr, path))
        else:
            if return_hists:
                if len(tmp) == 3:  # charness, hists, hash
                    return tmp[0], tmp[1]
                else:
                    success = False  # charness, hash
            elif return_names:
                if len(tmp) == 3:  # charness, scenelet_names, hash
                    return tmp[0], tmp[1]
                else:
                    success = False  # charness, hash
            else:
                return tmp[0]

    if not success:
        dmat = scipy.io.loadmat(path)
        pose_charness = np.squeeze(dmat['pigraph_pose_charness']).astype('f4')
        success = True

    if return_hists:
        hists = parse_charness_histograms(dmat)
        if not success:
            pickle.dump((pose_charness, hists, hash_curr), open(path_pickle, 'wb'), -1)
        return pose_charness, hists
    elif return_names:
        scenelet_names = [str(n[0][0]) for n in dmat['pigraph_scenelet_names']]
        if not success:
            pickle.dump((pose_charness, scenelet_names, hash_curr),
                           open(path_pickle, 'wb'), -1)
        return pose_charness, scenelet_names
    else:
        if not success:
            pickle.dump((pose_charness, hash_curr), open(path_pickle, 'wb'), -1)
        return pose_charness


class SceneletChoice:
    def __init__(self, input_id, scenelet_id, distance, charness=1.):
        self.input_id = input_id
        self.scenelet_id = scenelet_id
        self.distance = distance
        self.charness = charness

    def __repr__(self):
        return "SceneletChoice(%d, %d, dist %f, charness %f)" \
               % (self.input_id, self.scenelet_id, self.distance, self.charness)


def select_match_for_each_frame(
    dists, thresh_distance, strategy, blacklist=None, pose_charness=None,
    thresh_charness=None, path_video=None, name_video=None, is_augment=False):
    """

    :param dists: size: m_video_scenelets x n_pigraph_scenelets.
    :param thresh_distance:
    :param strategy:
    :param blacklist:
    :param pose_charness:
    :param thresh_charness:
    :param path_video:
    :return:
    """
    assert strategy in frozenset({0, 1}), "Two strategies only"

    #
    # Show distribution
    #
    fig = plt.figure(figsize=(16, 8))
    if name_video is not None:
        plt.suptitle("Input scene: %s" % name_video)
    # lg.info("pose_charness: %s" % pose_charness)
    ax0 = fig.add_subplot(221)
    ax0.hist(pose_charness, bins=100)
    ylim0 = ax0.get_ylim()
    ax0.set_title("Database characteristicness distribution (%d scenelets)"
                  % pose_charness.size)
    # draw threshold line
    ax0.plot((thresh_charness, thresh_charness), (0, ylim0[1]), 'r-')
    # ax0.set_ylim(ylim[0], ylim[1])
    best_per_frame = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(dists.shape[0]), best_per_frame]

    ax1 = fig.add_subplot(222)
    ax1.hist(min_dists, bins=100)
    ylim1 = ax1.get_ylim()
    ax1.set_title("Matching distance (%d matches)" % min_dists.size)
    ax1.plot((thresh_distance, thresh_distance), (0, ylim1[1]), 'r-')

    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax2.scatter(min_dists, pose_charness[best_per_frame])
    ax2.set_title("Cross plot")
    ax2.set_xlabel("matching distance")
    ax2.set_ylabel("characteristicness")
    ax2.set_ylim(0., 1.)
    ax2.set_xlim(-0.01, max(0.1, np.max(min_dists) + 0.01))
    ax2.plot((0., thresh_distance), (thresh_charness, thresh_charness), 'r-')
    ax2.plot((thresh_distance, thresh_distance), (thresh_charness, 1.), 'r-')
    plt.draw()
    if path_video is not None:
        path_fig = os.path.join(path_video, 'matching_distributions.jpg')
        plt.savefig(path_fig)
        lg.info("saved to %s" % path_fig)
        # os.system("(eog %s &)" % path_fig)

    #
    #  Strategy 0: top matches
    #
    if strategy == 0:
        mins = np.argsort(dists.ravel())
        mins_2d = np.unravel_index(mins, dists.shape)

    taken = set() if blacklist is None else set(blacklist)

    #
    # Strategy 1: best for each column, possibly with the same scenelet
    #
    if strategy == 1:
        best_per_frame = np.argmin(dists, axis=1)
        assert best_per_frame.shape == (dists.shape[0], ), \
            "No: %s, expected: %s" \
            % (repr(best_per_frame.shape), repr(dists.shape))

    # Do selection
    selection = []
    for frame_id in range(dists.shape[0]):
        first = None
        if strategy == 0:  # take the best
            try:
                first = \
                    next(
                        [input_id, scenelet_id]
                        for input_id, scenelet_id in zip(mins_2d[0], mins_2d[1])
                        if input_id == frame_id and scenelet_id not in taken
                        and (thresh_charness is None
                             or pose_charness[scenelet_id] > thresh_charness))
            except StopIteration:
                continue
        else:  # strategy 1 - with overlap
            best_indices = np.argsort(dists[frame_id, :])
            assert dists[frame_id, best_indices[0]] \
                <= dists[frame_id, best_indices[-1]], \
                "Wrong order: %s >= %s" \
                % (dists[frame_id, best_indices[0]],
                   dists[frame_id, best_indices[-1]])
            if is_augment:
                candidates = [
                    (frame_id, best_indices[lin_id])
                    for lin_id in range(len(best_indices))
                    if pose_charness[best_indices[lin_id]] > thresh_charness
                       and dists[frame_id, best_indices[lin_id]] < thresh_distance
                ]
                if not len(candidates):
                    lg.info("Could not find match for frame %d" % frame_id)
                    continue
                probabilities = np.array([
                    thresh_distance - dists[e[0], e[1]] for e in candidates
                ])
                # lg.debug("probs: %s,\ncands:\n%s" % (probabilities, candidates))
                probabilities /= np.sum(probabilities)
                id_first = np.random.choice(a=len(candidates), p=probabilities)
                first = candidates[id_first]
            else:
                try:
                    first = next(
                        (frame_id, best_indices[lin_id])
                        for lin_id in range(len(best_indices))
                        if pose_charness[best_indices[lin_id]] > thresh_charness
                        and dists[frame_id, best_indices[lin_id]] < thresh_distance
                    )
                except StopIteration:
                    lg.info(
                        "Didn't find anything for frame %d, because "
                        "best has distance %g >= %g thresh "
                        "and charness %g >= %g thresh"
                        "\n\tworst has distance %g >= %g thresh "
                        "and charness %g >= %g thresh"
                        % (frame_id,
                           dists[frame_id, best_indices[0]], thresh_distance,
                           pose_charness[best_indices[0]], thresh_charness,
                           dists[frame_id, best_indices[-1]], thresh_distance,
                           pose_charness[best_indices[-1]], thresh_charness))
                    continue
        if first is None:
            lg.error("first is None, it should not happen")
            continue

        l = list(taken)
        l.append(first[1])
        taken = set(l)

        scenelet_choice = SceneletChoice(
            input_id=first[0],
            scenelet_id=first[1],
            distance=dists[first[0], first[1]])
        if pose_charness is not None:
            scenelet_choice.charness = \
                pose_charness[scenelet_choice.scenelet_id]

        selection.append(scenelet_choice)

    # selection = sorted(selection,
    #                    key=(lambda x: x.distance)
    #                    if pose_charness is None
    #                    else (lambda x: 1. - x.charness))
    selection = sorted(selection, key=lambda x: x.distance)
    if not len(selection):
        raise NotEnoughMatchesError("No matches found! Decrease thresholds?")
    lg.info("First and last selected: %s..%s"
                 % (repr(selection[0]), repr(selection[-1])))
    # selection = \
    #     np.array(selection,
    #              dtype={'names': ['video_id', 'scenelet_id', 'distance'],
    #                    'formats': ['i4', 'i4', 'f4']})
    return selection


def pos_and_fw_nrmlzd(skeleton, frame_id, tr_ground=None):
    """Fetch position and normalized forward of skeleton at time"""
    p = skeleton.get_pose(frame_id)[:, Joint.PELV]
    if tr_ground is not None:
        fw = gm.normalized(gm.project_vec(
            skeleton.get_forward(frame_id, estimate_ok=False),
            tr_ground))
    else:
        fw = gm.normalized(
            skeleton.get_forward(frame_id, estimate_ok=False),
            tr_ground)
    return p, fw


def compare_frames(skeleton_sclt, skeleton_in,
                   frame_id_sclt, frame_id_in, tr_ground):
    p_skel, fw_skel = pos_and_fw_nrmlzd(skeleton_in, frame_id_in, tr_ground)
    p_sclt, fw_sclt = pos_and_fw_nrmlzd(skeleton_sclt, frame_id_sclt, tr_ground)

    # p_skel = skeleton_in.get_pose(frame_id_in)[:, Joint.PELV]
    # p_sclt = skeleton_sclt.get_pose(frame_id_sclt)[:, Joint.PELV]
    translation = gm.project_vec(p_skel - p_sclt, tr_ground)
    # lg.info("translation between\n\t%s\n\t%s is\n\t%s"
    #              % (p_skel, p_sclt, translation))

    # fw_skel = skeleton_in.get_forward(frame_id_in, estimate_ok=False)
    # fw_skel = gm.normalized(gm.project_vec(fw_skel, tr_ground))
    #
    # fw_sclt = skeleton_sclt.get_forward(frame_id_sclt, estimate_ok=False)
    # fw_sclt = gm.normalized(gm.project_vec(fw_sclt, tr_ground))

    # assert np.allclose(p_skel2, p_skel), "Wrong pos"
    # assert np.allclose(p_sclt2, p_sclt), "Wrong pos"
    # assert np.allclose(fw_skel2, fw_skel), "Wrong fw"
    # assert np.allclose(fw_sclt2, fw_sclt), "Wrong fw"

    # sys.stderr.write("Flipping forward by 180.\n")
    angle = gm.angle_3d(fw_sclt, fw_skel)
    # lg.info("angle between\n\t%s\n\t%s\n\t\tis %g (%g)"
    #              % (fw_sclt, fw_skel, angle, np.rad2deg(angle)))
    # lg.info("tr_ground:\n%s" % tr_ground)
    if np.cross(fw_sclt, fw_skel).dot(tr_ground[:3, 1]) < 0.:
        angle = -angle
    return translation.astype(np.float32), np.float32(angle)


def add_arrow(ax, start, dir, ec, minmax, arrow_scale=0.1, w=1.,
              arrow_head_mult=1., fc='y'):
    assert len(dir) == 3, "TODO: implement for 2D"
    assert len(start) == 3, "TODO: implement for 2D"

    dx = dir[2] * arrow_scale * (w + .1)
    dy = -dir[0] * arrow_scale * (w + .1)
    ax.arrow(start[2], -start[0], dx, dy, fc=fc, ec=ec,
             head_width=0.01 * arrow_head_mult)
    _min = (min(start[2], start[2] + dx),
            min(-start[0], -start[0] + dy))
    _max = (max(start[2], start[2] + dx),
            max(-start[0], -start[0] + dy))
    if minmax is not None:
        return ((min(minmax[0][0], _min[0]),
                 min(minmax[0][1], _min[1])),
                (max(minmax[1][0], _max[0]),
                 max(minmax[1][1], _max[1])))


def add_plot(ax, pos_3d, linestyle='-', minmax=None, **kwargs):
    ax.plot(pos_3d[:, 2], -pos_3d[:, 0],
            linestyle, linewidth=0.5, **kwargs)
    if minmax is not None:
        return ((min(minmax[0][0], np.min(pos_3d[:, 2])),
                 min(minmax[0][1], np.min(-pos_3d[:, 0]))),
                (max(minmax[1][0], np.max(pos_3d[:, 2])),
                 max(minmax[1][1], np.max(-pos_3d[:, 0]))))


def align_save(py_scenelet, py_video, tr_ground, path_out_root, vis=None,
               full_video=None, charness_histograms=None, show=True):
    lg.debug("Starting %s" % py_scenelet)
    # disk path
    out_dir = os.path.join(path_out_root, py_scenelet.name_scene)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sclt_cpy = copy.deepcopy(py_scenelet)

    frames_sc = py_scenelet.skeleton.get_frames()
    frames_in = py_video.skeleton.get_frames()
    n_frames_sc = len(frames_sc)
    n_frames_in = len(frames_in)
    assert n_frames_sc == n_frames_in, "Not same amount of frames? %s %s" \
        % (py_scenelet.skeleton.get_frames(), py_video.skeleton.get_frames())
    n_frames = min(n_frames_sc, n_frames_in)

    weighting = scipy.signal.get_window((
        'gaussian', np.float32(n_frames) / np.float32(6.)),
        Nx=n_frames).astype(np.float32)

    # apply confidence
    if py_video.confidence is not None:
        assert weighting.size == len(py_video.confidence), \
            "confidence is not correct size: %d %d" \
            % (weighting.size, len(py_video.confidence))
        for i, conf in enumerate(py_video.confidence):
            weighting[i] *= conf
    else:
        sys.stderr.write("Don't have confidence...")

    weighting_sum = np.sum(weighting)

    # per-frame comparison
    translations_angles = \
        [compare_frames(
            py_scenelet.skeleton, py_video.skeleton,
            frames_sc[lin_id], frames_in[lin_id], tr_ground)
            for lin_id in range(n_frames)]

    # translations
    ts = np.vstack((e[0] for e in translations_angles)) \
        * weighting[:, np.newaxis]
    translation = np.sum(ts, axis=0) / weighting_sum
    assert translation.shape == (3, ), \
        "Wrong shape: %s" % repr(translation.shape)

    #
    # rotations
    #

    angle_vec = np.sum(
        np.array(
            [
                [
                    np.float32(np.cos(e[1])) * weighting[i],
                    np.float32(np.sin(e[1])) * weighting[i]
                ]
                for i, e in enumerate(translations_angles)
            ],
            dtype='f4'),
        axis=0
    ) / weighting_sum
    angle = np.arctan2(angle_vec[1], angle_vec[0])

    #
    # rotate around mid frame first
    #
    mid_frame_sclt = sclt_cpy.skeleton.get_representative_frame()
    mid_frame_skel = py_video.skeleton.get_representative_frame()

    # rotate to align
    pos_sclt = sclt_cpy.skeleton.get_joint_3d(Joint.PELV, mid_frame_sclt)
    assert pos_sclt.dtype == np.float32, "Wrong type: %s" % pos_sclt.dtype
    pos_sclt = gm.project_vec(pos_sclt, tr_ground)
    assert pos_sclt.dtype == np.float32, \
        "Wrong type: %s, %s" % (pos_sclt.dtype, tr_ground.dtype)

    # compose transform
    tr = np.dot(
        gm.translation(translation + pos_sclt),
        np.dot(
            gm.rotation_matrix(angle, tr_ground[:3, 1], dtype=np.float32),
            gm.translation(-pos_sclt)
        )
    )

    # apply transform
    sclt_cpy.apply_transform(tr)

    #
    # Translate again (TODO: skip first translation)
    #

    del ts
    del translation
    del tr

    # per-frame comparison
    translations_angles2 = \
        [
            compare_frames(
                sclt_cpy.skeleton, py_video.skeleton,
                frames_sc[lin_id], frames_in[lin_id], tr_ground)
            for lin_id in range(n_frames)
        ]
    translations2 = np.vstack((e[0] for e in translations_angles2)) \
                    * weighting[:, np.newaxis]
    translation2 = np.sum(translations2, axis=0) / weighting_sum
    assert translation2.shape == (3, ), \
        "Wrong shape: %s" % repr(translation2.shape)

    transform2 = np.dot(
        tr_ground,
        gm.translation(translation2)
    )

    # apply transform
    sclt_cpy.apply_transform(transform2)

    #
    # Show
    #

    if show:
        fig = plt.figure(figsize=(36, 20), dpi=200)
        plt.gcf()
        # ax1 = plt.subplot2grid((2, 8), (0, 0), rowspan=2, aspect='equal')
        ax2 = plt.subplot2grid((3, 8), (0, 0), rowspan=1, colspan=2, aspect='equal')
        ax3 = plt.subplot2grid((3, 8), (1, 0), rowspan=2, colspan=2, aspect='equal')

        axes_h = [
            plt.subplot2grid((3, 8), (0, 2), rowspan=3, colspan=2, aspect='equal'),
            plt.subplot2grid((3, 8), (0, 4), rowspan=3, colspan=2, aspect='equal'),
            plt.subplot2grid((3, 8), (0, 6), rowspan=3, colspan=2, aspect='equal'),
        ]

        axes = [ax2, ax3]
        minmax = dict((ax, ((100., 100.), (-100., -100.)))
                      for ax in frozenset({ax2, ax3}))
        minmax[ax3] = add_plot(ax3, full_video.poses[:, :, Joint.PELV],
                               linestyle='x--k', minmax=minmax[ax3])
        fontsize = 14
        arrow_scale = .1

        # print("Up: %s" % tr_ground[:3, 1])
        # plt.suptitle("Final angle: %g" % np.rad2deg(angle))
        side = 1
        for lin_id, (translation, angle) in enumerate(translations_angles):
            w = weighting[lin_id]
            # get data
            p_skel, fw_skel = \
                pos_and_fw_nrmlzd(py_video.skeleton, frames_in[lin_id], tr_ground)

            p_sclt_orig, fw_sclt_orig = \
                pos_and_fw_nrmlzd(py_scenelet.skeleton, frames_sc[lin_id], tr_ground)

            p_sclt, fw_sclt = \
                pos_and_fw_nrmlzd(sclt_cpy.skeleton, frames_sc[lin_id], tr_ground)

            # plot original arrows
            # minmax[ax1] = add_arrow(
            #     ax1, p_sclt_orig, fw_sclt_orig, ec='g',
            #     minmax=minmax[ax1], arrow_scale=arrow_scale, w=w, fc='g')
            # ax1.arrow(p_sclt_orig[2],
            #           -p_sclt_orig[0],
            #           fw_sclt_orig[2] * arrow_scale * w,
            #           -fw_sclt_orig[0] * arrow_scale * w,
            #           fc='y', ec='g', head_width=0.01)

            # plot other arrows
            for ax in {ax2, ax3}:
                # aligned
                minmax[ax] = add_arrow(
                   ax, p_sclt, fw_sclt, ec='g', fc='g',
                   minmax=minmax[ax], arrow_scale=arrow_scale, w=w,
                   arrow_head_mult=1.2)
                # input video
                minmax[ax] = add_arrow(
                    ax, p_skel, fw_skel, ec='b', fc='b',
                    minmax=minmax[ax], arrow_scale=arrow_scale, w=w,
                arrow_head_mult=1.)

            # text
            # ax1.annotate(
            #     s="da: %.1f,\nw: %.1g" % (np.rad2deg(angle), weighting[lin_id]),
            #     xy=(p_sclt_orig[2], -p_sclt_orig[0]),
            #     xytext=(p_sclt_orig[2], -p_sclt_orig[0]),
            #     fontsize=fontsize)
            ax2.annotate(
                s="da: %.1f,\nw: %.1g" % (np.rad2deg(angle), weighting[lin_id]),
                xy=(p_skel[2], -p_skel[0]),
                xytext=(p_skel[2]+side*0.05, -p_skel[0]),
                fontsize=fontsize/2.)

            # Circles
            if lin_id == n_frames_sc // 2:
                # ax1.add_artist(plt.Circle((p_sclt_orig[2], -p_sclt_orig[0]), 0.01,
                #                           color='g', fill=False))
                ax2.add_artist(plt.Circle((p_sclt[2], -p_sclt[0]), 0.01,
                                          color='g', fill=False))
                ax3.add_artist(plt.Circle((p_sclt[2], -p_sclt[0]), 0.01,
                                          color='g', fill=False))

                for ax in axes_h:
                    add_arrow(
                       ax, p_sclt, fw_sclt, ec='g', fc='g',
                       minmax=None, arrow_scale=arrow_scale * 4., w=w,
                       arrow_head_mult=4.)

            if lin_id == n_frames_in // 2:
                ax2.add_artist(plt.Circle((p_skel[2], -p_skel[0]), 0.01,
                                          color='b', fill=False))
                ax3.add_artist(plt.Circle((p_skel[2], -p_skel[0]), 0.01,
                                          color='b', fill=False))
            # side *= -1

        # paths
        assert py_scenelet.skeleton.poses is not sclt_cpy.skeleton.poses, "NO!"
        assert py_scenelet.skeleton is not sclt_cpy.skeleton.poses, "NO!"
        # minmax[ax1] = add_plot(
        #     ax1, pos_3d=py_scenelet.skeleton.poses[:, :, Joint.PELV],
        #     linestyle='x--g', minmax=minmax[ax1])
        minmax[ax2] = add_plot(
            ax2, pos_3d=sclt_cpy.skeleton.poses[:, :, Joint.PELV],
            linestyle='x--g', minmax=minmax[ax2])
        minmax[ax2] = add_plot(
            ax2, pos_3d=py_video.skeleton.poses[:, :, Joint.PELV],
            linestyle='x--b', minmax=minmax[ax2])

        if vis is not None:
            VisSkeleton.vis_skeleton(
                vis, sclt_cpy.skeleton.get_pose(mid_frame_sclt), "skel_sclt_transf",
                color_add=(-0.2, 0., 0.),
                forward=sclt_cpy.skeleton.get_forward(mid_frame_sclt,
                                                      estimate_ok=False))
            vis.show()

        # ax1.set_title("Scenelet unaligned\nCharacteristicness: %.3g"
        #               % py_scenelet.charness, fontsize=fontsize)
        # ax1.legend(["Scenelet"],
        #            fontsize=fontsize*0.75, loc=2, bbox_to_anchor=(-0.1, 1.))
        ax2.set_title(
           "Video and scenelet aligned (zoom)\nMatching distance: %.3g"
           % sclt_cpy.match_score)
        ax2.legend(["Scenelet", "Video"],
                   fontsize=fontsize*0.75, loc=2, bbox_to_anchor=(-0.1, 1.))
        ax3.set_title("Video and scenelet aligned (full scene)")
        ax3.legend(["Video"],
                   fontsize=fontsize*0.75, loc=2, bbox_to_anchor=(-0.1, 1.))
        for ax in axes:
            ax.set_xlim(minmax[ax][0][0] - 0.1, minmax[ax][1][0] + 0.1)
            ax.set_ylim(minmax[ax][0][1] - 0.1, minmax[ax][1][1] + 0.1)
            # lg.debug("lims: %s" % repr(minmax[ax]))

        if charness_histograms is not None:
            cmap = matplotlib.cm.get_cmap('jet')
            norm = matplotlib.colors.Normalize(vmin=0., vmax=0.4)
            transform = sclt_cpy.skeleton.get_transform_from_forward(
                dim=2, frame_id=-1)
            hist = charness_histograms[
                (sclt_cpy.name_scene, sclt_cpy.name_scenelet)]
            for ax_id, label in enumerate(['table', 'couch', 'chair']):
                ax = axes_h[ax_id]
                for e0, e1 in product(range(1, hist.edges[0].size),
                                      range(1, hist.edges[1].size)):
                    rect = \
                        np.array([
                            [hist.edges[0][e0-1], hist.edges[1][e1-1]],
                            [hist.edges[0][e0-1], hist.edges[1][e1]],
                            [hist.edges[0][e0], hist.edges[1][e1]],
                            [hist.edges[0][e0], hist.edges[1][e1-1]]
                        ])
                    centroid = np.mean(rect, axis=0)
                    arr = np.dot(transform[:2, :2], rect.T).T
                    arr += np.expand_dims(transform[:2, 2], axis=0)
                    arr = np.vstack((arr[:, 1], -arr[:, 0])).T
                    poly = geom.Polygon(arr)
                    # val = np.sum(hist._volume[label][e0, e1, :])
                    thetas = np.linspace(
                       (hist.edges[2][1] - hist.edges[2][0]) / 2.,
                       hist.edges[2][-1], num=hist._volume[label].shape[-1],
                       endpoint=False)
                    vals = [
                        hist.get_value(
                           np.array((centroid[0], centroid[1], theta)), label)
                        for theta in thetas
                    ]
                    val = np.max(vals)
                    # assert np.isclose(val, np.sum(vals)), \
                    #     "No: %s (=%g)\n %s (=%g)" \
                    #     % (hist._volume[label][e0, e1, :], val, vals, np.sum(vals))
                    # lg.debug("e0: %d, e1: %d, val: %g" % (e0-1, e1-1, val))
                    if val > 0.:
                        ax.annotate("%2.2f" % val,
                                    xy=(poly.centroid.x-0.05, poly.centroid.y-0.05),
                                    fontsize=8)
                    ax.add_artist(
                        PolygonPatch(poly, alpha=.4,
                                     facecolor=cmap(norm(val)), zorder=3))
                ax.set_xlim(minmax[ax3][0][0] - 0.25, minmax[ax3][1][0] + 0.25)
                ax.set_ylim(minmax[ax3][0][1] - 0.25, minmax[ax3][1][1] + 0.25)
                ax.set_title(label)

                add_plot(
                    ax, pos_3d=full_video.poses[:, :, Joint.PELV],
                    linestyle='--k')
            # lg.debug("transform:\n%s" % transform)

        # rect=left, bottom, right, top
        # plt.tight_layout(pad=1., h_pad=0.01, rect=(-0.05, -0.01, 1.05, 1.01))
        # fig.subplots_adjust(left=0.05, wspace=0.15, right=1.05)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
        plt.draw()
        # plt.show()

        # save by match score
        path_fig_dir = os.path.join(out_dir, os.pardir, "_figures_by_dist")
        if not os.path.exists(path_fig_dir):
            os.mkdir(path_fig_dir)
        plt.savefig(
            os.path.join(
                path_fig_dir,
                "%.6f_%s__%s_aligned_%3.2f.jpg"
                % (sclt_cpy.match_score, py_scenelet.name_scene,
                   py_scenelet.name_scenelet,
                   sclt_cpy.get_time(mid_frame_skel))))

        # save by frame
        path_fig_dir = os.path.join(out_dir, os.pardir, "_figures_by_time")
        if not os.path.exists(path_fig_dir):
            os.mkdir(path_fig_dir)
        plt.savefig(
            os.path.join(
                path_fig_dir,
                "%03.2f_%s__%s_aligned_%.6f.jpg"
                % (sclt_cpy.get_time(mid_frame_skel), py_scenelet.name_scene,
                   py_scenelet.name_scenelet,
                   sclt_cpy.match_score)))

        # deb_name_scene = 'ahouse_mati5_2014-05-16-19-45-44'
        # deb_name_sclt = 'skel_scenelet_56'
        # if py_scenelet.name_scene == deb_name_scene \
        #     and py_scenelet.name_scenelet == deb_name_sclt:
        #     plt.show()
        # plt.show()
        plt.close()

    #
    # Save
    #

    # align in time roughly (for display)
    frame_offset = mid_frame_skel - mid_frame_sclt
    sclt_cpy.skeleton.move_in_time(frame_offset)
    # make sure time is not negative
    for frame_ in (frame_ for frame_ in sclt_cpy.skeleton.get_frames()
                   if frame_ < 0):
        lg.info("Removing %d" % frame_)
        sclt_cpy.skeleton.remove_pose(frame_)
    lg.info("[integer time] Moved by %s to %s"
                 % (frame_offset, mid_frame_sclt + frame_offset))

    # copy continuous time from input video (align in time roughly v2)
    assert len(py_video.skeleton._times) == len(sclt_cpy.skeleton._times), \
        "?? %s %s" % (len(py_video.skeleton._times), len(sclt_cpy.skeleton._times))
    sclt_cpy.skeleton._times = py_video.skeleton._times

    # save
    sclt_cpy.save(
        os.path.join(out_dir, "%s_aligned_%03d"
                     % (py_scenelet.name_scenelet, mid_frame_skel)))
    lg.info("Saved %s, aligned to %s__%s"
                 % (sclt_cpy, py_video.name_scene, py_video.name_scenelet))


def compute_blacklist(names_scenelets, blacklist_names):
    return [lin_id
            for lin_id in range(len(names_scenelets))
            if names_scenelets[lin_id].split('__')[0].split('_')[0]
            in blacklist_names]

def get_empty_probs(best_indices, dists_best_inv, K, names_recording_scenelets,
                    charness_histograms, mask):
    # template = None
    # angles = None
    out = {}
    masked = np.ma.masked
    id_center = None
    cats = None
    for row in range(best_indices.shape[0]):
        sum_w = 0.
        angle_column = None
        for col in range(K):
            sclt_id = best_indices[row, col]
            if sclt_id is masked or (mask.ndim != 0
                                     and mask[row, col] == masked):
                lg.debug("skipping %d, %d" % (row, col))
                continue
            name_recording, name_scenelet_recording = Scenelet.to_old_name(
                names_recording_scenelets[sclt_id])
            h = charness_histograms[(name_recording, name_scenelet_recording)]
            weight = np.float32(dists_best_inv[row, col])
            assert weight is not masked, "no: %s" % weight

            if cats is None:
                cats = [cat for cat in sorted(h._samplers.keys(),
                                              key=lambda c: CATEGORIES[c])
                        if cat in CATEGORIES]
                assert all([CATEGORIES[c0] < CATEGORIES[c1]
                            for c0, c1 in zip(cats[:-1], cats[1:])]), \
                    "no: %s" % cats
                if len(cats) != len(h._samplers):
                    lg.warning("Missing a category? %s"
                               % set(h._samplers.keys()).differene(cats))
                id_center = tuple(s//2 for s in h._volume[cats[0]].shape[:2])
                assert id_center == (3, 3), "no: %s" % id_center
            tmp = np.array([h._volume[cat][id_center[0], id_center[1], 1:-1]
                            for cat in cats], dtype=np.float32)

            if angle_column is None:
                angle_column = weight * tmp
            else:
                angle_column += weight * tmp
            sum_w += weight

        if sum_w > 0.:
            out[row] = angle_column / sum_w

    return out

def get_best_indices_and_values(dists_masked):
    best_indices = np.ma.argsort(dists_masked, axis=1)
    # map to 1D coordinates
    offsets = np.arange(0, best_indices.shape[0])
    best_indices_lin = best_indices + (offsets * best_indices.shape[1])[:, None]
    # get best distances using 1D coordinats
    dists_best = np.take(dists_masked, best_indices_lin)
    return best_indices, dists_best


def calculate_charness_per_frame(
    dists: np.ndarray, pose_charness: np.ndarray,
    charness_histograms: Dict[str, SquareHistogram], thresh_distance: float,
    thresh_charness: float, path_out_root: str,
    query_full: Scenelet, query_scenelets: List[Scenelet],
    names_recording_scenelets: Dict[int, Dict[int, str]],
    recording_scenelets, tr_ground,
    K: int=5) -> Dict[int, np.float32]:
    """Estimate matching distance-weighted characteristicness for each
    center frame in the video scenelets.

    :param dists:
        2D array of descriptor distances.

        First index (N rows): id of video scenelet (fewer, ~100).

        Second index (M cols): id of database scenelets
        (more, e.g. ~3000).

        *shape*: (N, M)

    :param pose_charness:
        Characteristicness of database scenelet.

        *shape*: (M, )

    :param charness_histograms:
        Dictionary of database histograms.
    :param thresh_distance:
        Maximum distance to take scenelet match into account.
    :param thresh_charness:
        Minimum threshold for characteristicness.
    :param path_out_root:
        Output path (typically <video_path>/match).
    :param frame_ids:
        List of frame_ids that match the rows in dists.
    :param query_full:
        Scenelet containing the full query skeleton.
    :param query_scenelets:
        The scenelets with the true positions.
    :param K:
        How many matches to take into account when calculating per-pose
        characteristicness.

    :return: Dictionary with charness per frame.
    """
    thrdmult = 1.
    fontsize = 20
    fontsize_axis = fontsize * 0.9
    charness_masked = np.ma.masked_less_equal(pose_charness, thresh_charness,
                                              copy=True)
    dists_masked_c = np.ma.masked_where(
        np.tile(charness_masked.mask, (dists.shape[0], 1)), dists, copy=True)
    # dists_masked = np.ma.masked_greater(
    #     np.ma.compress_cols(dists_masked), thresh_distance, copy=False)
    # dists_masked = np.ma.masked_greater(
    #     dists_masked_c, thresh_distance, copy=False)
    dists_masked_d = np.ma.masked_greater(dists, thresh_distance * thrdmult,
                                          copy=True)
    dists_masked = np.ma.masked_array(
        dists, mask=np.logical_or(dists_masked_c.mask, dists_masked_d.mask)
    )
    # assert np.allclose(dists_masked.mask, dists_masked_alt.mask)

    # charness_masked = np.ma.compressed(charness_masked)

    # find best indices (puts masked values at back of each row)
    best_indices, dists_best = get_best_indices_and_values(dists_masked)
    # best_indices = np.ma.argsort(dists_masked, axis=1)
    # map to 1D coordinates
    # offsets = np.arange(0, best_indices.shape[0])
    # best_indices_2 = best_indices + (offsets * best_indices.shape[1])[:, None]
    # get best distances using 1D coordinats
    # dists_best = np.take(dists_masked, best_indices_2)
    # mask indices, where distance is masked, pick at most K
    best_indices = np.ma.masked_where(dists_best.mask, best_indices)[:, :K]
    # keep the corresponding distances only, for efficiency
    dists_best = dists_best[:, :K]
    # dists_best_d = np.take(dists_masked_d, best_indices_2)[:, :K]

    # NOTE: best_indices_2 is not valid anymore, it uses longer strides...
    # best_indices_2 = None

    # debug:
    # out = []
    # for row in range(best_indices.shape[0]):
    #     out.append([])
    #     for col in range(best_indices.shape[1]):
    #         idx = best_indices[row, col]
    #         if idx is np.ma.masked:
    #             continue
    #         v = dists_masked[row, idx]
    #         if v is not np.ma.masked:
    #             out[-1].append([row, idx])
    # for row in out:
    #     if len(row):
    #         lg.debug(list("(%g, %d, %d)" % (dists_masked[row_, col_], row_, col_)
    #                  for row_, col_ in row))
    # get characteristicness of closest
    charnesses = np.take(charness_masked, best_indices)  # [:, :K]

    # dists_best = np.take(dists_masked, best_indices_2)[:, :K]
    dists_best_inv = thresh_distance - dists_best
    charness_per_frame = np.sum(np.multiply(charnesses, dists_best_inv),
                                axis=1) \
                         / np.sum(dists_best_inv, axis=1)
    for row in range(charnesses.shape[0]):
        s = 0
        w = 0
        for k in range(K):
            ch = charnesses[row, k]
            dist = dists_best_inv[row, k]
            s += dist * ch
            w += dist

        if s is not np.ma.masked:
            assert np.isclose(s / w, charness_per_frame[row], rtol=1.5), \
            "%g/%g=%g, %g" % (s, w, s / w, charness_per_frame[row])


    best_indices_d, dists_best_d = get_best_indices_and_values(dists_masked_d)
    best_indices_d = best_indices_d[:, :K]
    dists_best_d = dists_best_d[:, :K]
    dists_best_inv_d = thresh_distance - dists_best_d
    non_empty_probs = get_empty_probs(
        best_indices_d, dists_best_inv_d, K, names_recording_scenelets,
        charness_histograms, dists_masked_d.mask)

    assert len(query_scenelets) == charness_per_frame.shape[0], "No"
    times = [
        query_sclt.get_time(query_sclt.skeleton.get_representative_frame())
        for query_sclt in query_scenelets
    ]
    charness_poses = {
        time: charness
        for time, charness in zip(times, charness_per_frame)
        if charness is not np.ma.masked
    }

    lin_ids = list(range(len(query_scenelets)))

    #
    # save to a new scenelet with just the middle frames as sample points
    #

    sclt_charness = copy.deepcopy(query_full)
    # erase skeleton info
    sclt_charness.skeleton = Skeleton()
    lg.warning("TODO: charness into skeleton instead of scenelet")
    sclt_charness.charness_poses = dict()
    # TODO: take delta_time into account in dest_lin_id
    non_empty_prob_shape = non_empty_probs[next(iter(non_empty_probs))].shape
    non_empty_probs_out = {'times': [],
                           'frame_ids': [],
                            'positions': [],
                           'non_empty_probs': []}

    # def add_frame_id(sclt_charness, non_empty_probs_out,
    #                  dest_lin_id, frame_id, charness, source_sclt):
    #     sclt_charness0
    # NOTE: "frame_ids" in the query_full skeleton are called
    # "time" in the query scenelets.
    # "time" in the query skeleton is ignored by the MATLAB code
    # generating the query scenelets.
    # query_full_times = query_full.skeleton.get_frames()
    # assert isinstance(query_full_times, list), \
    #     "Assumed a sorted list of integer ids."
    # query_full_times_lin_id = 0
    def add_entry_to(non_empty_probs_out, dest_lin_id, pos, time, non_empty_probs):
        if len(non_empty_probs_out['frame_ids']):
            assert non_empty_probs_out['frame_ids'][-1] != dest_lin_id, \
                "Double dest id: %d "
        non_empty_probs_out['frame_ids'].append(dest_lin_id)
        non_empty_probs_out['positions'].append(pos.tolist())
        non_empty_probs_out['times'].append(time)
        if non_empty_probs is not None:
            non_empty_probs_out['non_empty_probs'].append(
                non_empty_probs.tolist())
        else:
            non_empty_probs_out['non_empty_probs'].append(
                np.zeros(non_empty_prob_shape,
                         dtype=np.float32).tolist())

    lin_ids_custom = [lin_ids[0]] + lin_ids + [lin_ids[-1]]
    lg.debug("lin_ids: %s" % lin_ids_custom)
    dest_lin_id = 0
    for i, lin_id in enumerate(lin_ids_custom):
        query_sclt = query_scenelets[lin_id]
        is_special = i in (0, len(lin_ids_custom)-1)
        if is_special:
            lg.debug("is_special: %s" % repr((i, lin_id)))
        charness = charness_per_frame[lin_id] \
            if not is_special else np.ma.masked

        if i == 0:
            frame_id = query_sclt.skeleton.get_frames()[0]
        elif i == len(lin_ids_custom)-1:
            frame_id = query_sclt.skeleton.get_frames()[-1]
        else:
            frame_id = query_sclt.skeleton.get_representative_frame()
        time_ = query_sclt.get_time(frame_id)

        sclt_charness.skeleton.copy_frame_from(
            frame_id=frame_id, other=query_sclt.skeleton,
            may_overwrite=False, dest_frame_id=dest_lin_id)
        sclt_charness.skeleton.set_time(frame_id=dest_lin_id,
                                        time=time_)

        # always save position
        add_entry_to(non_empty_probs_out,
                     dest_lin_id=dest_lin_id,
                     pos=sclt_charness.skeleton.get_centroid_3d(dest_lin_id),
                     time=time_,
                     non_empty_probs=non_empty_probs[lin_id]
                     if (lin_id in non_empty_probs and not is_special)
                     else None)
        # non_empty_probs_out['times'].append(time_)
        # non_empty_probs_out['frame_ids'].append(dest_lin_id)
        # non_empty_probs_out['positions'].append(
        #     sclt_charness.skeleton.get_centroid_3d(dest_lin_id).tolist())

        # save emptiness prob
        # if lin_id in non_empty_probs:
        #     non_empty_probs_out['non_empty_probs'].append(
        #         non_empty_probs[lin_id].tolist())
        # else:
        #     non_empty_probs_out['non_empty_probs'].append(
        #         np.zeros(non_empty_prob_shape,
        #                  dtype=np.float32).tolist())

        # save charness, if exists
        if charness is not np.ma.masked:
            sclt_charness.set_charness_pose(frame_id=dest_lin_id,
                                            charness=charness)

        dest_lin_id += 1
    sclt_charness.add_aux_info('non_empty_probs', non_empty_probs_out)

    #
    # Plotting
    #

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    name_scene = path_out_root.split(os.sep)[-2]
    fig.suptitle(name_scene, fontsize=fontsize)

    major_ticks = np.arange(0.1, .7, 0.1)
    minor_ticks = np.arange(0.1, .7, 0.05)

    color_norm = cmNormalize(vmin=0, vmax=K)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='coolwarm')
    colors = [scalar_map.to_rgba(i) for i in range(K)]

    # I. Charness - Distance

    ax = axes[0, 0]
    handles = []
    for k in reversed(range(K)):  # colorcode series
        color = colors[k] if k < len(colors) else (0., 0., 0., 1.)
        ax.scatter(dists_best[:, k].ravel(), charnesses[:, k].ravel(),
                   c=color)
        handles.append(mpatches.Patch(color=color, label="#%s" % (k + 1)))
    ax.set_title("Closest %d matches per query scenelet" % K,
                 fontsize=fontsize)
    ax.set_ylabel('Characteristicness', fontsize=fontsize_axis)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which='both')
    ax.set_xlim(0., 0.105)
    ax.set_xlabel('Feature distance', fontsize=fontsize_axis)
    ax.legend(handles=list(reversed(handles)))

    # II. Weighted charness over time

    ax = axes[1, 0]
    ax.plot(times, charness_per_frame, 'x--')
    ax.set_title("Weighted characteristicness", fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize_axis)
    ax.set_ylabel('Characteristicness', fontsize=fontsize_axis)

    # III. Weighted charness in space

    ax = axes[1, 1]
    vmin = np.min(charnesses)
    vmax = np.max(charnesses)
    color_norm = cmNormalize(vmin=0, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='jet')
    add_plot(ax, query_full.skeleton.poses[:, :, Joint.PELV], linestyle='--')
    frame_ids = sclt_charness.skeleton.get_frames()
    entries = np.zeros((len(frame_ids), 3), dtype=np.float32)
    for lin_id, frame_id in enumerate(frame_ids):
        # query_sclt = query_scenelets[lin_id]
        pos = sclt_charness.skeleton.get_joint_3d(
            joint_id=Joint.PELV,
            frame_id=frame_id
        )
        charness = sclt_charness.get_charness_pose(frame_id)
        # charness = charness_per_frame[lin_id]
        # if charness is not np.ma.masked:
        #     entries[lin_id, :] = (pos[2], -pos[0], charness)
        entries[lin_id, :] = (pos[2], -pos[0], charness)

    cax = ax.scatter(entries[:, 0], entries[:, 1], c=entries[:, 2],
                     norm=color_norm, cmap=cm.jet, zorder=5)
    cbar = fig.colorbar(cax, ax=ax, norm=color_norm, cmap='jet', pad=0.01)
    ax.set_title("Pelvis path (%d/%d poses)"
                 % (charness_per_frame.count(), len(lin_ids)),
                 fontsize=fontsize)
    ax.set_xlabel('x', fontsize=fontsize_axis)
    ax.set_ylabel('z', fontsize=fontsize_axis)
    ax.set_aspect('equal')

    # IV. Sampled characteristic

    if False:
        ax = axes[1, 2]
        add_plot(ax, query_full.skeleton.poses[:, :, Joint.PELV], linestyle='--')

        weights = charness_per_frame / np.sum(charness_per_frame)
        size_ = min(len(lin_ids), 32)
        chosen = np.random.choice(lin_ids, size=size_,
                                  replace=False,
                                  p=weights)
        for lin_id in chosen:
            query_sclt = query_scenelets[lin_id]
            pos = query_sclt.skeleton.get_joint_3d(
                joint_id=Joint.PELV,
                frame_id=query_sclt.skeleton.get_representative_frame()
            )
            charness = charness_per_frame[lin_id]
            ax.scatter(pos[2], -pos[0], c=charness, cmap=cm.jet, norm=color_norm,
                       zorder=5)

            p_sclt, fw_sclt = \
                pos_and_fw_nrmlzd(query_sclt.skeleton,
                                  query_sclt.skeleton.get_representative_frame(),
                                  None)
            add_arrow(
                ax, p_sclt, fw_sclt, ec='g', fc='k',
                minmax=None, arrow_scale=.25, w=1.,
                arrow_head_mult=1.2)
        ax.set_title("Sampled 32")


    # V. Distance - Time

    ax = axes[0, 2]
    handles = []
    for k in reversed(range(K)):
        color = colors[k] if k < len(colors) else (0., 0., 0., 1.)
        ax.scatter(times, dists_best[:, k],
                   c=color, alpha=0.75)
        handles.append(mpatches.Patch(color=color, label="#%s" % (k+1)))
    ax.legend(handles=list(reversed(handles)))
    # ax.scatter(np.tile(frame_ids, (dists_best.shape[1], 1)),
    #            dists_best)
    ax.set_title("Closest distances", fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize_axis)
    ax.set_ylabel('Distance', fontsize=fontsize_axis)

    # VI. Charness - Time

    ax = axes[0, 1]
    handles = []
    for k in reversed(range(K)):
        color = colors[k] if k < len(colors) else (0., 0., 0., 1.)
        ax.scatter(times, charnesses[:, k].ravel(),
                   c=color, alpha=0.75)
        handles.append(mpatches.Patch(color=color, label="#%s" % (k+1)))
    ax.legend(handles=list(reversed(handles)))
    ax.set_title("Charnesses vs time", fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize_axis)
    ax.set_ylabel('Characteristicness', fontsize=fontsize_axis)

    p = os.path.abspath(
        os.path.join(path_out_root, os.pardir, "charness_%s.jpg" % name_scene))
    fig.savefig(p, dpi=300)
    plt.close()
    lg.debug("Saved to %s" % p)

    #
    # Emptiness
    #

    vmax = 0.
    X = []
    Y = []
    C = {'table': [], 'chair': [], 'couch': [], 'all': []}
    for pos, probs in zip(non_empty_probs_out['positions'],
                          non_empty_probs_out['non_empty_probs']):
        X.append(pos[2])
        Y.append(-pos[0])
        for cat in C:
            # mx = np.max(probs[CATEGORIES[cat]])
            if cat != 'all':
                mx = np.max(probs[CATEGORIES[cat]])
                C[cat].append(mx)
                if mx > vmax:
                    vmax = mx
        mx = np.sum([np.max(probs[c]) for c in range(len(probs))])
        C['all'].append(mx)
        vmax = max(vmax, mx)
    color_norm = cmNormalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(12, 12))
    for i, cat in enumerate(C):
        if cat != 'all':
            ax = plt.subplot2grid((4, len(C) - 1), (0, i), aspect='equal')
        else:
            ax = plt.subplot2grid((4, len(C) - 1), (1, 0),
                                  colspan=3, rowspan=3, aspect='equal')
        add_plot(ax, np.mean(query_full.skeleton.poses[:, :, :],  axis=-1),
                 linestyle='--')
        # for frame_id_ in query_full.skeleton.get_frames():
        #     pos_ = query_full.skeleton.get_joint_3d(Joint.PELV, frame_id=frame_id_)
            # ax.annotate("%d" % frame_id_, xy=(pos_[2], -pos_[0]))

        if cat != 'all':
            cax = ax.scatter(X, Y, c=C[cat], cmap=cm.jet, vmax=vmax)
            fig.colorbar(cax, ax=ax, cmap='jet', pad=0.01, fraction=0.05)
            ax.set_title("%s non empty probability" % cat)
        else:
            unthresholded_min_dists = np.min(dists, axis=1).tolist()
            unthresholded_min_dists.insert(0, 1.)
            unthresholded_min_dists.append(1.)
            cax = ax.scatter(X, Y, c=unthresholded_min_dists, cmap=cm.jet, vmax=vmax)
            fig.colorbar(cax, ax=ax, cmap='jet', pad=0.01, fraction=0.05)
            ax.set_title("Unthresholded min dists")
            for x, y, d in zip(X, Y, unthresholded_min_dists):
                ax.annotate("%.3f" % d,
                            xy=(x, y))

        # ax.set_title("%s non empty probability" % cat)
    plt.suptitle("Distance threshold %g" % (thresh_distance * thrdmult))
    plt.subplots_adjust(wspace=0.5)
    p = os.path.abspath(
        os.path.join(path_out_root, os.pardir, "emptiness_%s_%g.jpg" % (name_scene, thresh_distance * thrdmult)))
    fig.savefig(p, dpi=300)
    # plt.show()

    #
    #
    #
    fig = plt.figure()
    ax = fig.add_subplot(121)
    for lin_id in lin_ids:
        query_sclt = query_scenelets[lin_id]
        a = list(query_sclt.skeleton._times.values())
        lg.debug("a: %s" % a)
        b = np.repeat(lin_id + 1, len(query_sclt.skeleton._times))
        lg.debug("b: %s" % b)
        plt.plot(list(query_sclt.skeleton._times.values()), b)
    plt.plot(list(query_full.skeleton._times.values()),
             np.repeat(0, len(query_full.skeleton._times)))
    ax.set_title('Video length and query scenelet lengths (time)')
    ax.set_xlabel('time')
    ax.set_ylabel('query scenelet id')
    ax = fig.add_subplot(122, aspect='equal')
    add_plot(ax, np.mean(query_full.skeleton.poses[:, :, :],  axis=-1),
             linestyle='--', alpha=0.7)
    add_plot(ax, np.mean(query_scenelets[-1].skeleton.poses, axis=-1),
             linestyle='x-')
    p2d = non_empty_probs_out['positions'][-1]
    ax.scatter(p2d[2], -p2d[0],
               c=np.max(non_empty_probs_out['non_empty_probs'][-1]),
               cmap='jet', s=30, zorder=5, alpha=0.7)
    ax.set_title('Last query scenelet and its charness')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    # plt.show()

    # debug all matches
    if False:
        for lin_id in lin_ids:
            query_sclt = query_scenelets[lin_id]
            for k in range(best_indices_d.shape[1]):
                sclt_id = best_indices_d[lin_id, k]
                name_recording, name_scenelet_recording = Scenelet.to_old_name(
                    names_recording_scenelets[sclt_id])
                recording_scenelet = \
                    recording_scenelets[name_recording][name_scenelet_recording]

                p = os.path.abspath(
                    os.path.join(path_out_root, os.pardir, 'match_walk'))
                align_save(
                    py_scenelet=recording_scenelet,
                    py_video=query_sclt,
                    tr_ground=tr_ground,
                    path_out_root=p,
                    vis=None,
                    full_video=query_full.skeleton,
                    charness_histograms=charness_histograms,
                    show=False
                )
                break

    # sys.exit(0)

    return sclt_charness


def _work(id_row, selection, names_video_scenelets, input_video_scenelets,
          names_recording_scenelets, recording_scenelets, charness_histograms,
          skeleton_scenelet, transformation_ground, path_out_root, show):
    """

    :param id_row: Entry id in \p selection.
    :param selection: List of SceneletChoice objects
        assigning recording_scenelets to video_scenelets.
    :param names_video_scenelets:
    :param input_video_scenelets:
    :param names_recording_scenelets:
    :param recording_scenelets:
    :param charness_histograms:
    :param skeleton_scenelet:
    :param transformation_ground:
    :param path_out_root:
    :param show:
    :return:
    """
    lg.debug("Starting %d/%d" % (id_row, len(selection)))

    # Names
    name_video, name_scenelet_video = Scenelet.to_old_name(
       names_video_scenelets[selection[id_row].input_id])
    name_recording, name_scenelet_recording = Scenelet.to_old_name(
       names_recording_scenelets[selection[id_row].scenelet_id])

    # Input
    try:
        input_video_scenelet = \
            input_video_scenelets[name_video][name_scenelet_video]
    except KeyError:
        lg.error("MISSING %s %s" % (name_video, name_scenelet_video))
        return

    # Scenelet
    try:
        recording_scenelet = \
            recording_scenelets[name_recording][name_scenelet_recording]
    except KeyError:
        lg.error("MISSING %s %s" % (name_recording, name_scenelet_recording))
        return

    # copy attributes
    recording_scenelet.charness = selection[id_row].charness
    recording_scenelet.match_score = selection[id_row].distance

    # Align
    align_save(
       py_scenelet=recording_scenelet,
       py_video=input_video_scenelet,
       tr_ground=transformation_ground,
       path_out_root=path_out_root,
       vis=None,
       full_video=skeleton_scenelet.skeleton,
       charness_histograms=charness_histograms,
       show=show
    )

def get_query_frame_ids_from_query_scenelets(
    names_query: List[str], query_scenelets: Dict[str, Dict[str, Scenelet]]
) -> List[int]:
    """Get's the representative frame_id for each query video scenelet,
    that is each row in the distance matrix.


    :param names_query:
        The unique name of each query scenelet in the distance matrix.

    :param query_scenelets:
        The generated query scenelets.
    """

    # prepare output
    frame_ids = []
    # for each entry (ordered)
    for id_row, name_query_new in enumerate(names_query):
        # convert to original name of scene and scenelet
        name_query, name_scenelet = Scenelet.to_old_name(name_query_new)
        # get representative (middle) frame of query scenelet
        frame_ids.append(
            query_scenelets[name_query][name_scenelet].skeleton
                .get_representative_frame()
        )
    assert len(frame_ids) == len(names_query), "Something's wrong"
    assert all(f0 <= f1 for f0, f1 in zip(frame_ids[:-1], frame_ids[1:])), \
        "Not in order?\n%s" \
        % [(f0, f1)
           for f0, f1 in zip(frame_ids[:-1], frame_ids[1:])
           if f0 >= f1]

    return frame_ids

def unroll_scenelets(
    names_query: List[str], query_scenelets: Dict[str, Dict[str, Scenelet]]
) -> List[Scenelet]:
    """Sorts the scenelets into a list, where their linear id matches the
        linear id (row) in the dists matrix.


    :param names_query:
        The unique name of each query scenelet in the distance matrix.

    :param query_scenelets:
        The generated query scenelets.
    :return: Ordered list of scenelets.
    """

    # prepare output
    out = []
    # for each entry (ordered)
    for id_row, name_query_new in enumerate(names_query):
        # convert to original name of scene and scenelet
        name_query, name_scenelet = Scenelet.to_old_name(name_query_new)
        # get representative (middle) frame of query scenelet
        out.append(query_scenelets[name_query][name_scenelet])
    assert len(out) == len(names_query), "Something's wrong"

    return out

def main_candidates(argv=None):
    np.set_printoptions(linewidth=200, suppress=True)

    # Parsing
    parser = argparse.ArgumentParser(
        "Generate scenelet candidates based on input pose sequence")
    parser.add_argument(
        "path_skeleton",
        help="Path to skel_ours.json or skel_denis_opt.json")
    parser.add_argument(
        "-thresh-dist", dest='thresh_dist', type=np.float32, required=True,
        help="Distance threshold (Default: 0.1)")
    parser.add_argument(
        "-d", dest="path_scenelets", required=True)
    parser.add_argument(
        "--out-limit", help="How many output scenelets", default=-1, type=int
    )
    parser.add_argument(
        "-thresh-charness", default=0.5, type=np.float32,
        help="Characteristicness threshold. 1. is most characteristic, "
             "0. is not charactersitic at all"
    )
    parser.add_argument(
       "-no-show", help="Save visualizations...", action="store_true"
    )
    parser.add_argument(
       '--no-mt', help="Disable multithreading", action='store_true'
    )
    parser.add_argument(
       '--augment', help="Augmentation by not choosing the best match",
       action="store_true"
    )
    mode_distance = 'gaussian'
    args = parser.parse_args(argv if argv is not None else sys.argv)
    args.show = not args.no_show
    # assert not args.show, "args: %s" % args
    mode_scenelet = args.path_scenelets.partition('__')[-1]

    path_video = os.path.dirname(args.path_skeleton)
    name_video = os.path.split(path_video)[-1]
    # print("path_video: %s\n name_video is %s" % (path_video, name_video))
    path_video_scenelets = \
        os.path.abspath(os.path.join(path_video, os.pardir,
                                     "video_scenelets__%s" % mode_scenelet))
    if not os.path.exists(path_video_scenelets):
        path_video_scenelets = os.path.abspath(os.path.join(
           path_video, os.pardir, os.pardir,
           "video_scenelets__%s" % mode_scenelet))
        if not os.path.exists(path_video_scenelets):
            assert False, "Could not find %s" % path_video_scenelets

    name_distances = "distance__%s_large.mat" % mode_distance
    name_charness_scenelets = "charness__%s.mat" % mode_distance

    _name_out_root = os.path.dirname(args.path_skeleton)

    # Output directory (with cleanup)
    _path_out_root = os.path.join(_name_out_root, "match")
    if os.path.exists(_path_out_root):
        for d in os.listdir(_path_out_root):
            if os.path.isdir(os.path.join(_path_out_root, d)):
                shutil.rmtree(os.path.join(_path_out_root, d))

    # Read distances
    path_distances = \
        os.path.join(path_video_scenelets, name_video, name_distances)
    lg.info("Reading\n\t%s" % split_path(path_distances))
    dists, names_scenelets, names_query = read_distances(path_distances)

    # Read pose charness
    path_charness = \
        os.path.join(args.path_scenelets, name_charness_scenelets)
    lg.info("Reading\n\t%s" % split_path(path_charness))
    pose_charness, charness_histograms = read_charness(
        path_charness, return_hists=True)

    # Read ground transform
    path_query_full = os.path.join(path_video, "skel_%s.json" % name_video)
    query_full = \
        Scenelet.load(path_query_full)
    tr_ground = np.asarray(query_full.aux_info['ground'], dtype='f4') \
        if 'ground' in query_full.aux_info \
        else np.identity(4, dtype='f4')

    blacklist = compute_blacklist(names_scenelets, blacklist_names={})


    # get selection
    lg.info("Matching...")
    selection = select_match_for_each_frame(
       dists, thresh_distance=args.thresh_dist,
       strategy=1,  # best with overlap
       pose_charness=pose_charness,
       blacklist=blacklist, thresh_charness=args.thresh_charness,
       path_video=path_video,
       name_video=name_video,
       is_augment=args.augment
    )
    # limit size
    if 0 < args.out_limit < len(selection):
        selection = selection[:args.out_limit]

    #
    # Load scenelets
    #

    # Database scenelets
    _path_scenes_pickle = os.path.abspath(
        os.path.join(args.path_scenelets, os.pardir, 'recordings.pickle'))
    if os.path.exists(_path_scenes_pickle):
        lg.info("Reading scenelets from pickle: %s" % _path_scenes_pickle)
        _py_recordings = pickle_load(open(_path_scenes_pickle, 'rb'))
    else:
        lg.info("Reading scenelets from %s..." % args.path_scenelets)
        _py_recordings = read_scenelets(args.path_scenelets)
        for name_recording, name_scenelet, scenelet \
                in get_scenelets(_py_recordings):
            scenelet.skeleton.compress_time()
        pickle.dump(_py_recordings, open(_path_scenes_pickle, 'wb'))
    lg.info("len(py_recordings): %s" % len(_py_recordings))

    # Input scenelets
    path_query_scenelets = \
        os.path.abspath(os.path.join(path_video_scenelets, name_video))
    path_queries_pickle = \
        os.path.abspath(os.path.join(path_query_scenelets, os.pardir,
                                     "%s_inputs.pickle" % name_video))
    if os.path.exists(path_queries_pickle):
        lg.debug("Reading queries from pickle: %s" % path_queries_pickle)
        query_scenelets = pickle.load(open(path_queries_pickle, 'rb'))
    else:
        lg.info("Reading input scenelets from %s" % path_query_scenelets)
        query_scenelets = read_scenelets(path_query_scenelets)
        for _, _, scenelet_query in get_scenelets(query_scenelets):
            scenelet_query.center_time()
        pickle.dump(query_scenelets, open(path_queries_pickle, 'wb'))
    lg.debug("n query_scenelets: %s" % len(query_scenelets))
    assert len(query_scenelets), \
        "No input scenelets: %s" % len(query_scenelets)

    #
    # Estimate per-pose charness
    #

    if query_full.charness_poses and len(query_full.charness_poses):
        lg.warning("Overwriting charness? %s"
                   % query_full.charness_poses)
    query_scenelets_list = unroll_scenelets(names_query, query_scenelets)
    sclt_charness = calculate_charness_per_frame(
        dists, pose_charness, charness_histograms,
        thresh_distance=args.thresh_dist,
        thresh_charness=args.thresh_charness,
        path_out_root=_path_out_root,
        query_full=query_full,
        query_scenelets=query_scenelets_list,
        names_recording_scenelets=names_scenelets,
        recording_scenelets=_py_recordings,
        tr_ground=tr_ground
    )
    # query_full.charness_poses = charness_poses
    path_out_sclt_charness = os.path.join(path_video,
                                          "skel_%s-charness.json" % name_video)
    sclt_charness.save(path=path_out_sclt_charness)
    lg.debug("Saved to %s" % path_out_sclt_charness)

    # vis = Visualizer()
    # vis_input(_py_recordings)

    # Align
    lg.info("Starting alignment...")

    if args.no_mt:
        for row in range(len(selection)):
            _work(row, selection, names_query, query_scenelets, names_scenelets,
                  _py_recordings, charness_histograms, query_full,
                  tr_ground, _path_out_root, args.show)
    else:
        raise RuntimeError("MT will be much slower on python3 as well")
        processes = []
        pool = multiprocessing.Pool(4)
        for row in range(len(selection)):
            processes.append(pool.apply_async(
               _work, (row, selection, names_query, query_scenelets, names_scenelets,
                  _py_recordings, charness_histograms, query_full,
                  tr_ground, _path_out_root, args.show)
               # {
               #     'names_query': names_query,
               #     'selection': selection,
               #     'names_scenelets': names_scenelets,
               #     'query_scenelets': query_scenelets,
               #     '_py_recordings': _py_recordings,
               #     'tr_ground': tr_ground,
               #     '_path_out_root': _path_out_root,
               #     'query_full': query_full,
               #     'charness_histograms': charness_histograms,
               #     'show': False}
            ))
        lg.debug("enlisted...closing")
        pool.close()
        pool.join()
        lg.debug("closed, getting...")
        [proc.get() for proc in processes]

    path_state_pickle = os.path.abspath(
        os.path.join(_name_out_root, 'state.pickle'))
    if os.path.exists(path_state_pickle):
        lg.debug("Deleting state pickle: %s" % path_state_pickle)
        os.remove(path_state_pickle)


if __name__ == '__main__':
    main_candidates(sys.argv)
