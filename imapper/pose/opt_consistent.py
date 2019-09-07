import argparse
import itertools
import os
import shutil
import sys
from math import ceil, floor
import copy
import cv2
import numpy as np
import tensorflow as tf
from functools import partial

from imapper.visualization.plotting import plt

from tensorflow.contrib.opt import ScipyOptimizerInterface

from imapper.config.conf import Conf
from imapper.logic.joints import Joint
from imapper.logic.scenelet import Scenelet
from imapper.pose.config import INPUT_SIZE
from imapper.pose.loss_intersection import \
    create_oo_intersection_losses, create_intersection_jo_losses
from imapper.pose.match_gap import \
    find_gaps, read_scenelets as read_orig_scenelets
from imapper.pose.parse_scenelet_db import is_close, plot_charness
from imapper.pose.save_consistent import export_scenelet
from imapper.pose.tf_occlusion_v1 import PairedOcclusionV1
from imapper.pose.tf_variables_manager import TFVariablesManager
from imapper.pose.unk_manager import PoseSource, UnkManager, _FTYPE_NP
from imapper.pose.visualization.show_opt3 import show_output
from imapper.util.my_argparse import argparse_check_exists
from imapper.util.json import json
from imapper.util.my_os import listdir
from imapper.util.my_pickle import pickle, pickle_load
from imapper.util.stealth_logging import lg
from imapper.util.timer import Timer
from imapper.util.my_os import makedirs_backed

import pdb
# import pydevd

# _FTYPE = tf.float64


class Weights(object):
    def __init__(self, proj=1., smooth=0.1, isec_oo=0.01, occl=25.):
        self._proj = _FTYPE_NP(proj)
        self._smooth = _FTYPE_NP(smooth)
        self._isec_oo = isec_oo
        """Object-object intersection weight."""
        self._isec_jo = _FTYPE_NP(isec_oo * 0.5)
        """Joint-object intersection weight."""
        self._occl = _FTYPE_NP(occl)

    @property
    def proj(self):
        return self._proj

    @property
    def reproj(self):
        """This is an alias for `proj`"""
        return self._proj

    @property
    def isec_jo(self):
        return self._isec_jo

    @property
    def isec_oo(self):
        return self._isec_oo

    @property
    def smooth(self):
        return self._smooth

    @property
    def occl(self):
        return self._occl


def read_scenelets(d_scenelets, dist_thresh=1., limit=0):
    pjoin = os.path.join

    # get full pigraph scenes
    p_scenelets = [pjoin(d_scenelets, f)
                   for f in os.listdir(d_scenelets)
                   if f.startswith('skel') and f.endswith('.json')]

    out = []
    for p_scenelet in p_scenelets:
        sclt = Scenelet.load(p_scenelet)
        skeleton = sclt.skeleton
        oids_to_remove = [oid
                          for oid, scene_obj in sclt.objects.items()
                          if not is_close(scene_obj, skeleton, dist_thresh)]
        for oid in oids_to_remove:
            sclt.objects.pop(oid)
        out.append(sclt)
        if limit != 0 and len(out) >= limit:
            break

    return out


def print_loss(o_loss_reproj, o_smooth_diffs, o_loss_isec_jo,
               o_loss_isec_oo, o_loss, o_loss_occl, weights):
    wlp = weights.proj * np.sum(o_loss_reproj)
    wls = weights.smooth * np.sum(np.abs(o_smooth_diffs))
    wlijo = weights.isec_jo * max(o_loss_isec_jo, 0.)
    wlioo = weights.isec_oo * max(o_loss_isec_oo, 0.)
    wlo = weights.occl * max(o_loss_occl, 0.)
    loss = wlp + wls + wlijo + wlioo + wlo
    lg.info(
      "loss:\n"
      "\treproj\t{lp:g} \t* {wp:g} = \t {wlp:g}\n"
      "\tsmooth\t{ls:g} \t* {ws:g} = \t {wls:g}\n"
      "\tjoiobj\t{lijo:g} \t* {wijo:g} = \t{wlijo:g}\n"
      "\tobjobj\t{lioo:g} \t* {wioo:g} = \t{wlioo:g}\n"
      "\toccl  \t{lo:g} \t* {wo:g} = \t{wlo:g}\n"
      "= {wlp:g} + {wls:g} + {wlijo:g} + {wlioo:g} + {wlo:g}\n"
      "= {loss:g} =?= {o_loss:g}".format(
        wp=weights.proj, lp=np.sum(o_loss_reproj),
        ws=weights.smooth, ls=np.sum(np.abs(o_smooth_diffs)),
        wijo=weights.isec_jo, lijo=o_loss_isec_jo,
        wioo=weights.isec_oo, lioo=o_loss_isec_oo,
        wo=weights.occl, lo=o_loss_occl,
        wlp=wlp, wls=wls, wlijo=wlijo, wlioo=wlioo, wlo=wlo,
        loss=loss, o_loss=o_loss
      ))
    if not np.isclose(o_loss, loss):
        lg.error("Loss not accurate: %g vs. %g output"
                 % (o_loss, loss))


def plot_rectangle_3d(ax, poly, normals=False):
    wrapped = np.concatenate((poly, poly[0:1, :]), axis=0)
    ax.plot(wrapped[:, 0], wrapped[:, 2], wrapped[:, 1])
    if normals:
        c = np.mean(poly, axis=0)
        for i in range(wrapped.shape[0]):
            e0 = wrapped[i, :] - wrapped[(i+1) % wrapped.shape[0], :]
            e0 /= np.linalg.norm(e0)
            e1 = wrapped[(i+1) % wrapped.shape[0], :] \
                 - wrapped[(i+2) % wrapped.shape[0], :]
            e1 /= np.linalg.norm(e1)
            a = np.cross(e0, e1) + c
            ax.plot([c[0], a[0]], [c[2], a[2]], [c[1], a[1]])


def create_occl_loss(pos_3d, polys_3d, joints_to_occlude,
                     visibility, pos_t_indices, dtype_np):
    """Deprecated"""
    #RayIntersector.ray_poly_intersection(pos_3d, )

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')

    # n_rects = 0
    # py_obj_vxs = []
    # py_obj_t_indices = []
    # for obj, idx_t, id_cat in scene_objects:
    #     for part in obj.parts.values():
    #         rect = part.obb.rectangles_3d().reshape((-1, 3)) - py_t[idx_t]
    #         py_obj_vxs.extend(rect)
    #         n_rects += 1
    #         py_obj_t_indices.extend([idx_t for _ in range(rect.shape[0])])
    #         assert len(py_obj_vxs) == len(py_obj_t_indices)
    # np_obj_vxs = np.array(py_obj_vxs, dtype=dtype_np)
    # np_obj_t_indices = np.array(py_obj_t_indices, dtype=np.int32)
    #
    # obj_vxs = tf.Variable(initial_value=np_obj_vxs,
    #                       trainable=False, name='occl_objs_vertices')
    # lg.debug("obj_vxs: %s" % obj_vxs)
    # # transformation indices for each object vertex
    # transform_indices = tf.constant(
    #   np_obj_t_indices[:, None], shape=[len(py_obj_t_indices), 1],
    #   name='occl_objs_transform_indices')
    # # transformations for object vertices
    # transforms_tiled = tf.gather_nd(transforms, indices=transform_indices,
    #                                 name="occl_objs_transforms")
    # obj_vxs_t = tf.squeeze(
    #   tf.matmul(transforms_tiled[:, :, :3], obj_vxs[:, :, None]), axis=-1
    # )
    # lg.debug("obj_vxs_t: %s" % obj_vxs_t)
    # obj_vxs_t += transforms_tiled[:, :, 3]
    # lg.debug("obj_vxs_t: %s" % obj_vxs_t)
    #
    # polys = tf.reshape(obj_vxs_t, (n_rects * 6, 4, 3))
    # poly_indices = tf.reshape(transform_indices, (n_rects * 6, 4))
    # lg.debug("obj_vxs_t: %s" % obj_vxs_t)

    joints = tf.concat(
      tuple(pos_3d[:, :, joint] for joint in joints_to_occlude),
      axis=0)
    pos_3d_transposed = tf.transpose(pos_3d, perm=(2, 0, 1))
    # joints2 = tf.gather(pos_3d_transposed, indices=joints_to_occlude)
    shp_vis = visibility.get_shape().as_list()
    assert len(shp_vis) == 2, "? %s" % repr(shp_vis)
    visibility_stacked = tf.concat(
      tuple(visibility[:, joint] for joint in joints_to_occlude),
      axis=0)
    distances = OcclusionV0.ray_poly_intersection(joints, polys_3d)
    lg.debug("distances: %s" % distances)
    lg.debug("visibility: %s" % visibility)
    mask = tf.less_equal(visibility_stacked, dtype_np(0.))
    fll = tf.fill(visibility_stacked.get_shape(), dtype_np(-1.))
    v_targets = tf.where(mask, fll, tf.negative(visibility_stacked))
    loss = tf.maximum(v_targets * distances, 0.)
    # loss = tf.where(distances < 0., loss, tf.zeros_like(loss))

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(0, 10)
    # ax.set_zlim(-5, 5)
    # plt.show()
    # sys.exit(0)
    deb = locals()
    return loss, deb


def loss_callback(loss, lp, d_oo, z_penalty, diff_p2d, packed_loss_grads,
                  packed_vars, translation, normalize_oo, occl_joints,
                  occl_polys, occl_loss, occl_distances, occl_visibility):
    sum_lp = np.sum(lp)
    sum_d_oo = np.sum(d_oo)
    lg.info("loss: %s, lp: %s, d_oo: %s, z_penalty: %s"
            % (loss, sum_lp, sum_d_oo, z_penalty))
    # lg.debug("packed_vars: %s" % packed_vars)
    # lg.debug("translation: %s" % translation)
    # lg.debug("packed_loss_grads: %s" % packed_loss_grads)
    lg.debug("normalize_oo: %s" % normalize_oo)
    # lg.debug("diff_p2d:\n%s" % diff_p2d)
    # lg.debug("d_oo:\n%s" % d_oo)
    assert not np.isnan(sum_lp), "lp:\n%s" % lp
    assert not np.isnan(sum_d_oo), "d_oo:\n%s" % d_oo
    assert not np.any(np.isnan(packed_loss_grads)), \
        "grads:\n%s" % packed_loss_grads
    n_poses = diff_p2d.shape[0]
    lg.debug("n_poses: %s" % n_poses)
    for frame_id in range(n_poses):
        l = occl_loss[n_poses]
        distance = occl_distances[frame_id]
        vis = occl_visibility[frame_id]
        if not (vis > 0. and distance < 0.):
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        for poly in occl_polys:
            plot_rectangle_3d(ax, poly)
        pnt = occl_joints[frame_id, :]
        ax.plot([0., pnt[0]],
                [0., pnt[2]],
                [0., pnt[1]])
        if l > 0.:
            s = "l: %g" % l
            ax.text(pnt[0], pnt[2], pnt[1], s=s, color='red')
            s = "d: %g" % distance
            ax.text(pnt[0], pnt[2], pnt[1]+1, s=s, color='blue')
            s = "v: %g" % vis
            ax.text(pnt[0], pnt[2], pnt[1]+2, s=s, color='k')
        lg.debug("pnt:\n%s" % pnt)
        # lg.debug("polys:\n%s" % occl_polys)
        np.save('/tmp/polys.npy', occl_polys)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 8)
        ax.set_zlim(-5, 5)
        plt.title("frame_id: %s" % frame_id)
        plt.show()


def compute_occl_loss_per_transform(d_per_joint_large, len_joints_to_occlude,
                                    loss_occl_per_pose, loss_occl_unn,
                                    n_poses_moving, n_transforms, session,
                                    um, independent,
                                    np_transform_indices_both):
    """

    Args:
        independent(bool):
            Scenelets don't interact, each joint has to be occluded by each
            transform (e.g. scenelet).
    """
    # dynamic poses
    o_loss_occl_per_pose, o_d_per_joint_w_empties, o_loss_occl_unn = \
        session.run([loss_occl_per_pose, d_per_joint_large, loss_occl_unn])
    if independent:
        assert o_loss_occl_per_pose.ndim == 2 and \
               o_loss_occl_per_pose.shape[0] == n_transforms+1, \
            "Expected a n_poses x n_transforms: %s" \
            % repr(o_loss_occl_per_pose.shape)
        assert np.all(o_loss_occl_per_pose < 1e5), "Empties left..."
        o_loss_occl_per_transform_unn = np.sum(o_loss_occl_per_pose,
                                               axis=1)
        assert o_loss_occl_per_transform_unn.shape[0] == n_transforms + 1, \
            "No: %s" % repr(o_loss_occl_per_transform_unn.shape)
        tids, counts = np.unique(np_transform_indices_both, return_counts=True)
        counts[:-1] += counts[-1]
        o_loss_occl_per_transform_normalizer = counts

        # normalize
        o_loss_occl_per_transform = o_loss_occl_per_transform_unn \
                                    / o_loss_occl_per_transform_normalizer
    else:
        selector = np.not_equal(um.transform_indices[:, None],
                                np.array(np.arange(0, n_transforms))[None, :])
        o_loss_occl_per_transform = np.tile(
          A=o_loss_occl_per_pose[:n_poses_moving, None],
          reps=(1, selector.shape[1]))
        o_loss_occl_per_transform_masked = np.ma.array(o_loss_occl_per_transform,
                                                       mask=selector)
        o_loss_occl_per_transform_unn = np.sum(o_loss_occl_per_transform_masked,
                                               axis=0)
        o_loss_occl_per_transform_normalizer = np.sum(np.logical_not(selector),
                                                      axis=0)
        # normalize
        o_loss_occl_per_transform = o_loss_occl_per_transform_unn \
                                    / o_loss_occl_per_transform_normalizer

        # static poses occluded by dynamic objects
        o_poly_inds = um.object_transform_indices.reshape((-1, 4))
        o_occluder_ids = np.argmin(o_d_per_joint_w_empties, axis=1).reshape(
          (-1, len_joints_to_occlude))
        occluder_ids = o_occluder_ids[n_poses_moving:, :]
        occluder_tids = np.take(a=o_poly_inds[:, 0], indices=occluder_ids, axis=0)
        accum = np.zeros((n_transforms, 2))
        for r in range(occluder_tids.shape[0]):
            for c in range(occluder_tids.shape[1]):
                tid = occluder_tids[r, c]
                accum[tid, 0] += o_loss_occl_unn[r, c]
                accum[tid, 1] += 1.
        msk = np.where(accum[:, 1] > 0)
        accum[msk, 0] /= accum[msk, 1]
        assert accum.shape[0] == o_loss_occl_per_transform.shape[0]
        o_loss_occl_per_transform += accum[:, 0]

    return o_loss_occl_per_transform


def get_d_per_joint_indices(um, pos_3d_both, n_poses_to_occlude,
                            len_joints_to_occlude,
                            w_static_occlusion):
    unique_pids, counts = np.unique(um.pids_polyids[:, 0],
                                    return_counts=True)
    counts = {k: v // 6
              for k, v in zip(unique_pids, counts)}
    if w_static_occlusion:
        assert len(counts) == pos_3d_both.shape[0], \
            "Expected entries for each pid: %s" % counts
    max_quads_per_joint = max(counts.values()) * 6
    lg.debug("max_quads_per_joint: %s" % max_quads_per_joint)
    d_per_joint_indices = []
    found_empty = []
    for pid in range(n_poses_to_occlude):  # pos_3d and not pos_3d_both
        try:
            _n_objs = counts[pid]
        except KeyError:
            lg.warning("Found a pid %d without objects %s"
                       % (pid, counts))
            found_empty.append(pid)
            continue
        for ob_id in range(_n_objs):
            for jid in range(len_joints_to_occlude):
                row = pid * len_joints_to_occlude + jid
                col0 = ob_id * 6
                d_per_joint_indices.extend(
                  [[row, col0 + poly_id] for poly_id in range(6)])
    if w_static_occlusion and len(found_empty) > 5:
        raise RuntimeError("Too many object-less pids found:\n%s,\n%s"
                           % (found_empty, counts))

    shp = [n_poses_to_occlude * len_joints_to_occlude,
           max_quads_per_joint]

    return d_per_joint_indices, max_quads_per_joint, shp


def get_d_per_joint_indices_indep(um, pos_3d_both, n_poses_to_occlude,
                                  len_joints_to_occlude,
                                  w_static_occlusion,
                                  np_moving_poly_ids_occluding,
                                  n_transforms,
                                  np_pids_transform_indices):
    unique_pids, counts = np.unique(um.pids_polyids[:, 0],
                                    return_counts=True)
    counts = {k: v // 6
              for k, v in zip(unique_pids, counts)}
    if w_static_occlusion:
        assert len(counts) == pos_3d_both.shape[0], \
            "Expected entries for each pid: %s" % counts
    max_quads_per_joint = max(counts.values()) * 6
    lg.debug("max_quads_per_joint: %s" % max_quads_per_joint)
    d_per_joint_indices = []
    found_empty = []

    # first pid that is static
    pid_static_first = um.pid_static_first

    # transform id of static poses
    shp0 = n_transforms
    if um.has_static():
        tid_static = um.tid_static
        shp0 += 1
    if pid_static_first is None:
        assert not w_static_occlusion, "Need the first pid if static occlusion"

    # linear index of the quad we are working with in the list
    # of distances
    i_poly = 0
    for pid in range(n_poses_to_occlude):  # pos_3d and not pos_3d_both
        try:
            _n_objs = counts[pid]
        except KeyError:
            lg.warning("Found a pid %d without objects %s"
                       % (pid, counts))
            found_empty.append(pid)
            continue

        poly_transform_inds = um.object_transform_indices.reshape((-1, 4))
        for ob_id in range(_n_objs):  # 0...Q
            for jid in range(len_joints_to_occlude):  # 0...14
                row = pid * len_joints_to_occlude + jid  # j = 0...J
                col0 = ob_id * 6  # q0, q6, q12

                # transform id of pose
                tid_pose = np_pids_transform_indices[pid, 0]

                # id of quad
                poly_id = np_moving_poly_ids_occluding[i_poly]

                # transform id of quad
                tid_poly = poly_transform_inds[poly_id, 0]

                if pid_static_first is not None:
                    assert tid_pose <= tid_static
                    assert tid_poly <= tid_static
                    is_pose_static = tid_pose == tid_static
                    is_poly_static = tid_poly == tid_static

                    if not (is_pose_static and is_poly_static):  # dyn-dyn
                        tid = tid_pose
                    elif is_pose_static:  # static pose - dynamic poly
                        assert not is_poly_static
                        tid = tid_poly
                    elif is_poly_static:  # dynamic pose - static poly
                        assert not is_pose_static
                        tid = tid_pose
                    else:  # static pose - static poly
                        assert is_pose_static and is_poly_static
                        assert tid_pose == tid_poly
                        tid = tid_pose

                #
                d_per_joint_indices.extend(
                  [[tid, row, col0 + side_id]
                   for side_id in range(6)])  # 0...6
                assert col0 + 5 < max_quads_per_joint
                i_poly += 6
                assert len(d_per_joint_indices) == i_poly
    if w_static_occlusion and len(found_empty) > 5:
        raise RuntimeError("Too many object-less pids found:\n%s,\n%s"
                           % (found_empty, counts))
    shp = [shp0, n_poses_to_occlude * len_joints_to_occlude,
           max_quads_per_joint]

    return d_per_joint_indices, max_quads_per_joint, shp


def sample_rotation(loss, loss_occl_per_transform, rotation,
                    loss_per_transform, session, n_steps):
    tmp_rot = rotation.eval()
    lg.debug("tmp_rot: %s" % tmp_rot.dtype)
    for dr in np.linspace(-np.pi/2., np.pi/2., n_steps):
        tmp_rot[:, :] = np.float32(dr)
        session.run(tf.assign(rotation, tmp_rot))
        o_loss, o_loss_occl_per_transform, o_loss_per_transform = session.run([
            loss, loss_occl_per_transform, loss_per_transform])
        lg.debug("\nloss:\t\t%g,\nloss_occl_per_tr:\t%s"
                 "\nloss_per_transform:\t%s"
                 % (o_loss, o_loss_occl_per_transform, o_loss_per_transform))

    sys.exit(0)


class OptParams(object):
    def __init__(self, gtol=1e-6, maxiter=100):
        self.gtol = gtol
        self.maxiter = maxiter


def match(query_full, d_query, query_2d_full, scenes_corresps, intrinsics,
          tr_ground, weights, is_scenes_transformed, opt_params,
          thresh_log_conf=Conf.get().path.thresh_log_conf,
          z_only=True, with_y=False, n_cands_per_batch=10,
          independent=False, with_occlusion=True,
          occlusion_eval_only=False, with_jo_intersection=False,
          cats_to_ignore=None, w_camera=True, w_static_occlusion=False,
          camera_init=(0., 0., 0.), d_padding=0.,
          dtype_np=_FTYPE_NP, silent=False, no_legs=False, min_poses=10):
    """

    Args:
      query_full (stealth.logic.scenelet.Scenelet):
        Initial 3D path (sparse, from LiftingFromTheDeep).
      d_query (str):
        Path on disk to the video scene.
      query_2d_full (stealth.logic.skeleton.Skeleton):
        All 2D keypoints from the video in xy.
      scenes_corresps (list):
        List of tuples that contain scenelets and their frame_id
        correspondences.
      intrinsics (np.ndarray): (3, 4)
        Camera intrinsics, assumed to only have fx, fy, cx, cy.
      tr_ground (np.ndarray): (4, 4)
        Room transform, only `ty` component is used.
      weights:
      thresh_log_conf (float):
        Deprecated.
      is_scenes_transformed (bool):
        Are the scenes in scenes_corresps already in room space?
      z_only (bool):
        Parametrize translations only in z direction (camera forward).
      with_y (bool):
        Optimize for full translation (x, y, z).
      n_cands_per_batch (int):
        How many scenelet candidates to output in independent mode.
      independent (bool):
        Should scenes interact (intersection, smoothness), and should we
        optimize for change of path from query_full.
      with_occlusion (bool):
        Should the loss contain the occlusion term.
      with_jo_intersection (bool):
        Should the loss contain the joint-object intersection term.
      cats_to_ignore (frozenset):
        Category labels that don't participate in the optimization.
        Typically `floor`, `book`, etc..
      w_camera (bool):
        Use camera transform.
      w_static_occlusion (bool):
        False: self occlusion only. True: occlude already placed path.
      camera_init (tuple):
        Initial camera extrinsics: rot_x, rot_y, t_y.
      d_padding (float):
        Joint-object intersection extra distance padding. If positive,
        it will make objects appear larger (e.g. account for skeleton
        extent). If negative, it will allow more intersection.
      dtype_np (DataType):
        The floating point type for optimization. Default: np.float64.
      silent (bool):
        Print less to console for speedup.
      no_legs (bool):
        Don't use object legs to save GPU memory.
    Returns:
        An ordered list of aligned scenelets.
    """

    # tensorflow data type
    dtype_tf = tf.as_dtype(dtype_np)

    pjoin = os.path.join
    assert independent or is_scenes_transformed, \
        "Consistent usually starts with scenes already in room space"

    if independent:
        if not occlusion_eval_only:
            lg.error("\n\nNOTO cclusion eval only?\n\n")
    elif occlusion_eval_only:
        lg.error("\n\nOcclusion eval only?\n\n")

    # PELV and NECK have NO confidence (set to 0 in main_denis.py) !!!
    # NOTE: this is used for 3D optimization, don't work with less than 14!
    joints_active = [Joint.RANK, Joint.RKNE, Joint.RHIP,
                     Joint.LHIP, Joint.LKNE, Joint.LANK,
                     Joint.THRX, Joint.HEAD, Joint.RWRI,
                     Joint.RELB, Joint.RSHO, Joint.LSHO,
                     Joint.LELB, Joint.LWRI]
    # joints_active = [Joint.RHIP, Joint.LHIP, Joint.HEAD]
    joints_remap = {j: i for i, j in enumerate(joints_active)}
    joints_to_smooth = (joints_remap[Joint.RHIP], joints_remap[Joint.LHIP])
    len_joints_to_occlude = len(joints_active)

    # 2D scaling from query_2d_full to scene space
    p_im = pjoin(d_query, 'origjpg', 'color_00001.jpg')
    assert os.path.exists(p_im), "Doesn't exist: %s" % p_im
    im_ = cv2.imread(p_im)
    shape_orig = im_.shape
    scale_2d = dtype_np(shape_orig[0] / dtype_np(INPUT_SIZE))

    if isinstance(w_camera, bool):
        w_camera = (w_camera, w_camera, w_camera)
    else:
        assert isinstance(w_camera, tuple) and len(w_camera) == 3, \
            "Expected three bools, whether to optimize camera rx, ry and ty."

    # sort based on video_frame_id
    # corresps:
    #     s_c[1] = [(frame_id_query, frame_id_scenelet), ( , ), ...]
    # first correspondence:
    #     s_c[1][0] = (frame_id_query, frame_id_scenelet)
    # video frame_id of first correspondence:
    #     s_c[1][0][0] = frame_id_query
    scenes, correspondences = zip(
      *sorted(scenes_corresps, key=lambda s_c: s_c[1][0][0])
    )

    # data manager instantiation
    um = UnkManager(thresh_log_conf=thresh_log_conf,
                    cats_to_ignore=cats_to_ignore, silent=silent)

    #
    # Moving scenes
    #

    assert len(correspondences) == len(scenes), \
        "Need same number: %s vs %s" % (len(correspondences), len(scenes))

    lg.info("[match] Started adding scenelets...")
    with Timer('adding scenelets'):
        # for each input scenelet with video-scenelet time correspondence
        for id_scene, (scene, corresp_frame_ids) \
                in enumerate(zip(scenes, correspondences)):

            # get next transform id
            idx_t = um.get_next_tid()

            # filter scenes with no valid objects
            obj_tid_catids = um.extract_scene_objects(scene, idx_t,
                                                      no_legs=no_legs)
            _min_poses = min_poses
            if not len(obj_tid_catids):
                lg.warning(
                    '[match] Too few objects, used to be skipping scene {:s}'
                    .format(scene.name_scenelet))
                
                # We are replacing outlier poses from LFD with shorter,
                # object-less scenelets from the database, even single
                # pose ones.
                lg.warning('[match] Overriding min_poses from {} to {}'
                           .format(min_poses, _min_poses))
                _min_poses = 1
                
                # continue

            # query_3d_full=no extra t_init if already good starting position
            added_scenelet = um.add_scenelet(
              corresp_frame_ids=corresp_frame_ids,
              query_2d_full=query_2d_full,
              query_3d_full=(query_full.skeleton
                             if independent else None),
              skeleton_scene=scene.skeleton, idx_t=idx_t,
              id_scene=id_scene, id_scene_part=id_scene,
              pose_source=PoseSource.SCENELET,
              tr_ground=tr_ground,
              min_poses=_min_poses)  # changed from 5
            # lg.error("REMOVE MIN POSES AFTER VNECT")

            # add objects only if pose correspondence successful
            if added_scenelet:
                um.add_scene_objects(obj_tid_catids)
    lg.info("[match] Finished adding scenelets...")

    if um.get_next_tid() == 0:
        lg.error("[match] No valid correspondences, returning False")
        return False

    #
    # Local poses
    #

    if independent:  # initial path is static
        for frame_id_query in query_full.skeleton.get_frames():
            # assert query_2d_full.has_pose(frame_id_query), \
            #     "We have a 3d local pose but no 2d detections??"
            if not query_2d_full.has_pose(frame_id_query):
                continue
            pose = query_full.skeleton.get_pose(frame_id_query)
            um.add_pose_static(pose=pose, frame_id_query=frame_id_query,
                               skeleton_2d=query_2d_full,
                               pose_source=PoseSource.LFD, id_scene=-2,
                               dtype_np=dtype_np)
    else:  # initial path added where we don't have scenelets
        frame_ids_singular = set(query_full.skeleton.get_frames()) \
            .difference(um.explained_frame_ids)
        for frame_id in frame_ids_singular:
            um.add_scenelet(
              corresp_frame_ids=[(frame_id, frame_id)],
              query_2d_full=query_2d_full,
              query_3d_full=None,
              skeleton_scene=query_full.skeleton,
              idx_t=um.get_next_tid(),
              id_scene=-1, id_scene_part=-1,
              tr_ground=tr_ground,
              min_poses=1,
              pose_source=PoseSource.LFD,
              dtype_np=dtype_np)

    # Create derived data in unknowns manager
    # TODO: optimize 2d and 3d object creation on read of Scenelet
    #       and resave to pickle
    um.finalize(scale_2d=scale_2d, intrinsics=intrinsics,
                with_intersection=with_jo_intersection or not independent,
                do_interact_other_groups=not independent,
                w_static_occlusion=w_static_occlusion,
                resolution_mgrid=Conf.get().path.mgrid_res * 2. if independent
                else Conf.get().path.mgrid_res,
                dtype_np=dtype_np)

    #
    # Work
    #
    lg.info("[match] Constructing graph...")
    n_transforms = um.translations.shape[0]  # TODO: check where this is # used,
    # and consider adding one after smoothness
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        if any(w_camera):
            # rx, ry, ty
            params_camera = tf.Variable(initial_value=camera_init,
                                        dtype=dtype_tf,
                                        trainable=True)
            # [[c1, 0, -s1]    [[1,  0 ,  0],    [[c1, s1s2, -s2c1],
            #  [0,  1,  0 ]  x  [0,  c0, s0],  =  [ 0,  c1,    s1 ],
            #  [s1, 0,  c1]]    [0, -s0, c0]]     [s2, -s1c2, c1c2]]
            cam_sn = tf.sin(params_camera[:2])
            cam_cs = tf.cos(params_camera[:2])

            transform_camera = tf.reshape(
              tf.stack(
                [cam_cs[1], cam_sn[0] * cam_sn[1], -cam_sn[1] * cam_cs[0], 0.,
                        0., cam_cs[0], cam_sn[0], params_camera[2],
                 cam_sn[1], -cam_sn[0] * cam_cs[1], cam_cs[0] * cam_cs[1], 0.,
                 0., 0., 0., 1.],
                axis=0),
              shape=(4, 4))

        # 3D translation
        translation_ = tf.Variable(initial_value=um.translations,
                                   name='initial_translations', dtype=dtype_tf)
        #assert with_y
        if with_y:
            translation = translation_
        elif z_only:
            translation = tf.concat((
                tf.Variable(initial_value=um.translations[:, 0:2] /
                                          um.translations[:, 2:3],
                            dtype=dtype_tf, trainable=False)
                        * translation_[:, 2:3],
                translation_[:, 2:3]), axis=1, name='translations')
        else:
            if not is_scenes_transformed:
                # scenelets need to be transformed from floor at 0
                t_y = tf.fill(dims=(um.translations.shape[0],),
                              value=tr_ground[1, 3].astype(dtype_np),
                              name='t_y')
            else:  # scenelets have already been aligned to ground
                t_y = tf.fill(dims=(um.translations.shape[0],),
                              value=dtype_np(0.),
                              name='t_y')
            translation = tf.concat(
                (translation_[:, 0:1], t_y[:, None], translation_[:, 2:3]),
                axis=1, name='translations'
            )

        # 3D rotation
        rotation = tf.Variable(um.rotations, name='rotation', dtype=dtype_tf)

        def select_poses(poses):
            return poses[:, :, joints_active]

        # visibility
        w = tf.Variable(
          np.squeeze(select_poses(um.confidence[:, None, :]), axis=1),
          trainable=False, name='w', dtype=dtype_tf)
        w_normalizer = tf.constant(
          value=np.reciprocal(np.sum(um.confidence)).astype(dtype_np),
          dtype=dtype_tf, shape=(), name='w_normalizer')

        # 2d feature points
        pos_2d_in = tf.Variable(select_poses(um.poses_2d), name='poses_2d',
                                trainable=False, dtype=dtype_tf)

        # 3d poses
        np_pos_3d_active = select_poses(um.poses_3d)
        n_poses = np_pos_3d_active.shape[0]
        n_poses_moving = um.pid_static_first or n_poses
        n_poses_static = n_poses - n_poses_moving
        assert n_poses_moving <= n_poses
        pos_3d_sclt = tf.Variable(np_pos_3d_active[:n_poses_moving, :, :],
                                  name='poses_3d',
                                  trainable=False, dtype=dtype_tf)

        # if len(um.poses_3d_static):
        #     pos_3d_static = tf.Variable(select_poses(um.poses_3d_static),
        #                                 name='poses_3d_static',
        #                                 trainable=False, dtype=dtype_tf)

        # assemble transformations
        transform_indices = tf.constant(
          um.transform_indices, name='transform_indices',
          shape=[len(um.transform_indices), 1], dtype=tf.int32)
        # transform_indices_1d = tf.squeeze(transform_indices, axis=1)

        # rotation around y
        my_zeros = tf.zeros((n_transforms, 1), dtype=dtype_tf,
                            name='my_zeros')
        my_ones = tf.ones((n_transforms, 1), dtype=dtype_tf, name='my_ones')
        c = tf.cos(rotation, name='cos')
        s = tf.sin(rotation, name='sin')
        transforms_world = tf.concat(
            [c, my_zeros, -s, translation[:, 0:1],
             my_zeros, my_ones, my_zeros, translation[:, 1:2],
             s, my_zeros, c, translation[:, 2:3],
             my_zeros, my_zeros, my_zeros, my_ones], axis=1)
        transforms_world = tf.reshape(transforms_world, (-1, 4, 4))

        if any(w_camera):
            transforms = tf.einsum('ij,ajk->aik',
                                   transform_camera,
                                   transforms_world)[:, :3, :]
        else:
            transforms = transforms_world[:, :3, :]

        # transform to 3d
        transform_tiled = tf.gather_nd(transforms, transform_indices)
        pos_3d = tf.add(
          x=tf.matmul(a=transform_tiled[:, :, :3],
                      b=pos_3d_sclt,
                      name='rotate_poses_3d'),
          y=transform_tiled[:, :, 3:4], name='poses_3d_transformed')

        # perspective divide
        pos_z = pos_3d[:, 2:3, :]
        mask_z_pos = tf.greater(pos_z, 0.1, name='mask_z_large_enough')
        pos_2d = tf.divide(
          pos_3d[:, :2, :],
          tf.where(mask_z_pos,
                   pos_z,
                   tf.fill(pos_z.get_shape(), dtype_np(0.001)),
                   name='where_z_large_enough'),
          name='perspective_divide'
        )

        # 2D residual
        diff_p2d = tf.subtract(pos_2d, pos_2d_in, name='diff_2d')

        # sqr of 2D residual: (u - u_0)^2 + (v - v_0)^2
        # diff_sqr = tf.square(diff_p2d[:, 0, :])
        #          + tf.square(diff_p2d[:, 1, :])
        diff_sqr = tf.reduce_sum(tf.square(diff_p2d[:, 0:2, :]), axis=1,
                                 name="diff_sqr")

        # weighted loss: w * ((u - u_0)^2 + (v - v_0)^2)
        masked_sqr = tf.multiply(diff_sqr, w[:n_poses_moving, ...],
                                 name='diff_sqr_masked')

        # reduce to loss
        loss_reproj_per_pose_unn = tf.reduce_sum(
          masked_sqr, axis=1, name='loss_reproj_per_pose_unn')
        loss_reproj_per_pose = tf.multiply(loss_reproj_per_pose_unn,
                                           w_normalizer,
                                           name='loss_reproj_per_pose')

        # penalty for being behind the camera
        # TODO: remove - from pos_z
        z_penalty = tf.nn.l2_loss(
          tf.where(mask_z_pos, tf.zeros_like(pos_z), -pos_z,
                   name='where_z_behind_camera'),
          name='z_penalty')

        loss_reproj = tf.add(x=loss_reproj_per_pose,
                             y=z_penalty,
                             name='loss_reproj_per_pose_plus_z_penalty')

        # compute per-scenelet loss for ranking
        if independent:
            # loss_reproj_per_transform = tf.Variable(
            #   initial_value=np.zeros(n_transforms),
            #   dtype=_FTYPE, name="loss_reproj_per_transform")

            sum_w_per_transform = tf.segment_sum(
              tf.reduce_sum(w[:n_poses_moving, :], axis=1),
              tf.squeeze(transform_indices, axis=-1),
              name='sum_w_per_transform')
            w_normalizer_per_transform = tf.reciprocal(
              sum_w_per_transform, name='w_normalizer_per_transform')

            # this can't be optimized for, but can be computed
            # hence loss_reproj goes into loss, and not this.
            loss_reproj_per_transform_unn = tf.segment_sum(
              loss_reproj_per_pose_unn,
              tf.squeeze(transform_indices, axis=-1),
              name='loss_reproj_per_transform_unn')

            loss_reproj_per_transform = tf.multiply(
              x=loss_reproj_per_transform_unn,
              y=w_normalizer_per_transform,
              name='loss_reproj_per_transform'
            )

        pos_3d_both = tf.concat(
          (pos_3d, tf.constant(np_pos_3d_active[n_poses_moving:, :, :])),
          axis=0,
          name='pos_3d_w_static')

        if um.has_static():
            np_transform_indices_static = np.ones(
              shape=(n_poses_static, 1), dtype=np.int32) * um.tid_static
            np_transform_indices_both = np.concatenate(
              (um.transform_indices[:, None],
               np_transform_indices_static),
              axis=0)
            transform_indices_both = tf.constant(
              value=np_transform_indices_both,
              name='transform_indices_both')
        else:
            transform_indices_both = transform_indices

        #
        # II. Spatial smoothing
        #

        actor_spans = query_full.skeleton.get_actor_empty_frames()
        assert len(actor_spans), "no"
        py_smooth_indices, py_smooth_indices_static, py_smooth_weights = \
            um.expand_to_smooth(joints_to_smooth=joints_to_smooth,
                                actor_spans=actor_spans)

        if len(py_smooth_indices):
            smooth_indices_moving = tf.constant(value=py_smooth_indices,
                                                dtype=tf.int32,
                                                name='smooth_indices_moving')
            # copy 3d positions to [[p3_0, p3_1], [...]_2]
            smooth_pairs_moving = tf.gather_nd(pos_3d_both,
                                               smooth_indices_moving,
                                               name="smooth_pairs_moving")
            # rotation -  added on 12/4/2018
            # smooth_rot = rotation[1:, ...] - rotation[:-1, ...]
        if len(py_smooth_indices_static):
            assert False, "This should be inactive"
            smooth_indices_static = tf.constant(value=py_smooth_indices_static,
                                                dtype=tf.int32,
                                                name='smooth_indices_static')
            # left of static is moving
            smooth_pairs_static_left = tf.gather_nd(
              pos_3d, smooth_indices_static[:, :3],
              name="smooth_pairs_static_left")
            smooth_pairs_static_right = tf.gather_nd(
              pos_3d_static, smooth_indices_static[:, 3:],
              name="smooth_pairs_static_right")
            smooth_pairs_static = tf.concat(
              (smooth_pairs_static_left, smooth_pairs_static_right), axis=1,
              name="smooth_pairs_static")

        smooth_weights = tf.Variable(initial_value=py_smooth_weights,
                                     trainable=False, dtype=dtype_tf,
                                     name='smooth_weights')
        # concatenate existing
        if len(py_smooth_indices) and len(py_smooth_indices_static):
            assert False
            smooth_pairs = tf.concat((smooth_pairs_moving,
                                      smooth_pairs_static),
                                     axis=0, name='smooth_pairs')
        elif len(py_smooth_indices):
            smooth_pairs = smooth_pairs_moving
        else:
            assert False
            assert len(py_smooth_indices_static)
            smooth_pairs = smooth_pairs_static

        # subtract positions: \sum_{xyz} (p3_0 - p3_1)^2
        smooth_diffs = tf.reduce_sum(
          tf.squared_difference(smooth_pairs[:, :3], smooth_pairs[:, 3:],
                                name='smooth_squared_difference'),
          axis=1, name='smooth_diffs')
        assert smooth_diffs.get_shape().as_list()[0] == \
            smooth_pairs.get_shape().as_list()[0], \
            "No: %s" % repr(smooth_diffs.get_shape().as_list())

        # weigh by time difference between positions
        smooth_diffs *= smooth_weights \
            * dtype_np(1. / len(py_smooth_weights))
        assert smooth_diffs.get_shape().as_list()[0] == \
            smooth_pairs.get_shape().as_list()[0] \
            and (len(smooth_diffs.get_shape().as_list()) == 1 or
                 smooth_diffs.get_shape().as_list()[1] == 1), \
            "No: %s" % repr(smooth_diffs.get_shape().as_list())
        # if True:
        #     loss_smooth = tf.square(smooth_diffs[1:] - smooth_diffs[:-1])
        #     with tf.Session(graph=graph) as session:
        #         lg.info("[match] variables_initializer")
        #         # Init
        #         session.run(tf.global_variables_initializer())
        #         o = smooth_weights.eval()
        #         sys.exit(0)
        #
        # else:
        loss_smooth = smooth_diffs

        #
        # III. Skeleton-object intersection
        #

        # (N, 14, 3) transpose to be able to flatten to 3D vertices
        # if len(um.poses_3d_static):
        #     pos_3d_both = tf.concat((pos_3d, pos_3d_static),
        #                             axis=0, name='pos_3d_both')
        assert transform_indices_both.shape[0] == pos_3d_both.shape[0]

        pos_3d_both_T = tf.transpose(a=pos_3d_both, perm=(0, 2, 1),
                                     name='pos_3d_both_T')

        tf_vars = TFVariablesManager()  # create 2D rectangles
        if with_jo_intersection or not independent:
            # TODO: loss_per_transform should be grouped by object
            with Timer('create_jo_intersection_losses'):
                # create top-view bounding rectangles
                tf_vars.create_objects_2d(transforms, um)
                loss_isec_jo, loss_isec_jo_per_transform, deb_jo = \
                    create_intersection_jo_losses(
                        pos_3d=pos_3d_both_T,
                        obj_2d_polys=tf_vars.obj_2d_polys,
                        joints_active=joints_active,
                        pos_3d_transform_indices=transform_indices_both,
                        obj_2d_poly_transform_indices=
                        tf_vars.obj_2d_transform_indices,
                        independent=independent, um=um,
                        d_padding=d_padding,
                        cat_ids_polys=tf_vars.cat_ids_polys)

            loss_isec_jo_sc = tf.reduce_sum(loss_isec_jo,
                                            name='loss_isec_jo_scalar')

        if not independent:
            # TODO: static object intersection
            with Timer('create_oo_intersection_losses'):
                if not tf_vars.is_objects_2d_created:
                    tf_vars.create_objects_2d(transforms, um)
                tf_vars.create_objects_2d_mgrids(transforms, um)
                loss_isec_oo, loss_oo_per_transform, deb_oo = \
                    create_oo_intersection_losses(
                      obj_2d_mgrid_vertices_transformed=
                      tf_vars.obj_2d_mgrid_vertices_transformed,
                      obj_2d_polys=tf_vars.obj_2d_polys,
                      oo_mask_interacting=tf_vars.oo_mask_interacting,
                      oo_mask_interacting_sum_inv=
                      tf_vars.oo_mask_interacting_sum_inv if
                      tf_vars._np_oo_sum > 1 else None)
            loss_isec_oo_sc = tf.reduce_sum(loss_isec_oo,
                                            name='loss_isec_oo_scalar')
            if loss_oo_per_transform is None:
                lg.warning("TODO: loss_oo_per_transform")


        #
        # IV. occlusion
        #

        polys_3d, polys_3d_transform_indices = \
            tf_vars.create_objects_3d(um, transforms)

        if with_occlusion:
            n_poses_to_occlude = pos_3d_both.shape[0]
            # (M,)
            moving_pids_to_occlude = tf.constant(
              value=um.pids_polyids[:, 0],
              dtype=tf.int64, name='moving_pose_ids_to_occlude')

            # (14 * M, 3)
            moving_poses_to_occlude = tf.reshape(
              tf.transpose(
                tf.reshape(
                  tf.gather(params=pos_3d_both_T,
                            indices=moving_pids_to_occlude,
                            name='moving_poses_to_occlude_2d'),  # (M, 14, 3)
                  shape=(-1, 6, 14, 3)),  # (M/6, 6, 14, 3)
                perm=(0, 2, 1, 3)),  # (M/6, 14, 6, 3)
              shape=(-1, 3), name='moving_poses_to_occlude_1d')

            #
            # gather polygons
            #
            # pids_polyids:
            #   pose_ids:   0      0           0       1         1
            #   poly_ids: 0...5, 6...11, ., k...k+5, 0...5, ., m...m+5
            # moving_poly_ids_occluding (14 * M,):
            #  pose_id:    0      0      0      0       0      0    1 ...
            #  joint_ids:  0      1  ... 13,    0   ... 13    13    0 ...
            #  poly_ids: 0...5, 0...5, 0...5, 6...11, 6...11, m+5, 0..5 ..
            # e.g. is expanded by the number of joints in groups of 6
            np_moving_poly_ids_occluding = \
                np.tile(A=um.pids_polyids[:, 1].reshape((-1, 1, 6)),
                        reps=(1, len_joints_to_occlude, 1)).flatten()
            moving_poly_ids_occluding = tf.constant(
              value=np_moving_poly_ids_occluding,
              dtype=tf.int32, name='moving_poly_ids_occluding'
            )

            # (14 * M, 4, 3)
            moving_polys_3d_occluding = tf.gather(
              params=polys_3d, indices=moving_poly_ids_occluding,
              name='moving_polys_3d_occluding'
            )

            # d gives one entry per pose and joint:
            # (14 * M, )
            d_bias = dtype_np(-10.)
            d = PairedOcclusionV1.point_quad_occlusion_distance(
              p=moving_poses_to_occlude,
              quads=moving_polys_3d_occluding,
              name='occlusion_distances')
            # lg.error("MULTIPLYING DISTANCE!!!")

            # make sure that all distances are positive when using
            # scatter_nd that is initialized to zeros
            # with tf.control_dependencies([
            #     tf.assert_greater(d, d_bias, name="distance_bias")
            # ]):
            d_offset = tf.subtract(d, d_bias, name='d_offset')

            if independent:
                d_per_joint_indices, max_quads_per_joint, shp_dpjo = \
                    get_d_per_joint_indices_indep(
                      um, pos_3d_both, n_poses_to_occlude,
                      len_joints_to_occlude, w_static_occlusion,
                      np_moving_poly_ids_occluding, n_transforms,
                      np_transform_indices_both)
            else:
                d_per_joint_indices, max_quads_per_joint, shp_dpjo = \
                    get_d_per_joint_indices(
                      um, pos_3d_both, n_poses_to_occlude,
                      len_joints_to_occlude, w_static_occlusion)

            d_per_joint_offset = tf.scatter_nd(
              indices=d_per_joint_indices,
              updates=d_offset,
              shape=shp_dpjo,
              name='d_per_joint_offset'
            )

            empty_value = dtype_np(1.e6)
            d_per_joint_large = tf.where(
              d_per_joint_offset > 0.,
              tf.add(d_per_joint_offset, d_bias),
              tf.ones_like(d_per_joint_offset) * empty_value,
              name='d_per_joint')

            # occluder_ids = tf.argmin(d_per_joint,
            #                          axis=1,
            #                          name='argmin_d_per_joint_1d')
            if independent:
                d_per_joint_w_empties = tf.reshape(
                  tf.reduce_min(d_per_joint_large,
                                axis=-1,
                                name='d_per_joint_1d'),
                  shape=(shp_dpjo[0], n_poses_to_occlude,
                         len_joints_to_occlude),
                  name='d_per_joint_2d_w_empties')
            else:
                d_per_joint_w_empties = tf.reshape(
                  tf.reduce_min(d_per_joint_large,
                                axis=1,
                                name='d_per_joint_1d'),
                  shape=(n_poses_to_occlude, len_joints_to_occlude),
                  name='d_per_joint_2d_w_empties')

            # zero out entries that were filled in automatically by scatter_nd
            d_per_joint = tf.where(d_per_joint_w_empties < empty_value - dtype_np(1.),
                                   d_per_joint_w_empties,
                                   tf.zeros_like(d_per_joint_w_empties),
                                   name='d_per_joint_2d')

            # if 'frame_ids_singular' in locals() and len(frame_ids_singular):
            #     visibilities_to_occlude = w[:-len(frame_ids_singular), :]
            # else:
            visibilities_to_occlude = w

            # # max(0, -sign(2*c-1) * sign(d) * (2*c-1)^2 * d^2
            # two_c_m1 = tf.subtract(
            #   x=tf.multiply(x=visibilities_to_occlude,
            #                 y=dtype_np(2.), name='two_c'),
            #   y=1., name='two_c_minus_1')

            c_m_half = tf.subtract(x=visibilities_to_occlude,
                                   y=dtype_np(0.5),
                                   name='c_minus_half')
            if independent:
                c_m_half = tf.tile(
                  input=tf.expand_dims(
                    input=c_m_half,
                    axis=0),
                  multiples=(shp_dpjo[0], 1, 1),
                  name="c_minus_half_tiled")

            # shape: (14 * N, )
            # loss_occl_unn = tf.maximum(
            #   tf.constant(0., dtype=dtype_tf),
            #   -tf.sign(two_c_m1) * tf.sign(d_per_joint)
            #   * tf.square(two_c_m1) * tf.square(d_per_joint),
            #   name='loss_occl_unnormalized')

            # (c - 0.5)^2 d^2,   if 0 < d, c < 0.5
            #         0      ,   otherwise
            loss_occl_unn = tf.where(
              tf.logical_and(tf.greater(d_per_joint, 0.,
                                        name='joint_not_occluded'),
                             tf.less(c_m_half, 0.,
                                     name="2d_low_conf"),
                             name='not_occluded_and_low_conf'),
              x=tf.square(tf.multiply(c_m_half, d_per_joint, name='c_half_d'),
                          name='c_half_d_squared'),
              y=tf.zeros_like(d_per_joint),
              name='loss_occl_unnormalized'
            )
            loss_occl = tf.multiply(
              loss_occl_unn,
              tf.reciprocal(tf.cast(tf.size(loss_occl_unn), dtype_tf),
                            name='loss_occl_normalizer'),
              name='loss_occl'
            )

            loss_occl_per_pose = tf.reduce_sum(loss_occl_unn, axis=-1,
                                               name='loss_occl_unn_per_pose')

            loss_occl_sc = tf.reduce_sum(loss_occl,
                                         name='loss_occl_scalar')

            # prepare accumulator
            # loss_occl_per_transform_unn = tf.Variable(
            #   initial_value=tf.zeros(n_transforms, dtype=loss_occl.dtype))
            # # accumulate
            # # shape: (n_transforms, )
            # # tmp = tf.squeeze(transform_indices[:-len(frame_ids_singular)], axis=-1) \
            # #     if 'frame_ids_singular' in locals() \
            # #     else tf.squeeze(transform_indices, axis=-1)
            # loss_occl_per_transform_unn = tf.scatter_add(
            #   ref=loss_occl_per_transform_unn,
            #   # TODO: switch to indices_both
            #   indices=tf.squeeze(transform_indices, axis=-1),
            #   updates=tf.reduce_sum(loss_occl_unn, axis=1,
            #                         name='loss_occl_unn_per_pose'),
            #   name='loss_occl_per_transform_unnormalized')
            #
            # # normalize
            # # TODO: append n_poses_static as an extra transform
            # _, n_poses_per_transforms = np.unique(
            #   transform_indices_both, return_counts=True)
            # assert (n_poses_per_transforms > 0).all(), "Empty transforms?"
            #
            # loss_occl_per_transform_normalizer = tf.constant(
            #   np.reciprocal(dtype_np(n_poses_per_transforms *
            #                          len_joints_to_occlude)),
            #   dtype=dtype_tf,
            #   name='loss_occl_per_transform_normalizer')
            # loss_occl_per_transform = tf.multiply(
            #   loss_occl_per_transform_unn,
            #   loss_occl_per_transform_normalizer,
            #   name='loss_occl_per_transform'
            # )

        #
        # Add up losses
        #

        loss_reproj_sc = tf.reduce_sum(loss_reproj, name='loss_reproj_scalar')
        loss_smooth_sc = tf.reduce_sum(loss_smooth, name='loss_smooth_scalar') \
                         # + tf.reduce_sum(smooth_rot, name='loss_smooth_rot_scalar') \
                         # / smooth_rot.get_shape().as_list()[0]

        loss = weights.proj * loss_reproj_sc \
               + weights.smooth * loss_smooth_sc

        lbfgs_options = {'gtol': opt_params.gtol, 'disp': True}
        if opt_params.maxiter > 0:
            lbfgs_options['maxiter'] = opt_params.maxiter

        if independent:  # independent
            loss_2 = loss
            if with_occlusion:
                if not occlusion_eval_only:
                    loss_2 += weights.occl * loss_occl_sc

            if with_jo_intersection:
                loss_2 += weights.isec_jo * loss_isec_jo_sc

            # TODO: + weights.isec_oo * loss_isec_oo_sc

            lg.error("add object-object loss to independent branch")

            # For scoring the candidates, cannot be optimized for, but can
            # be computed afterwards.
            loss_per_transform = weights.proj * loss_reproj_per_transform

            if with_jo_intersection:
                if transform_indices_both != transform_indices:
                    assert len(loss_isec_jo_per_transform.get_shape()
                               .as_list()) == 1, "Assumed 1D"
                    loss_per_transform += weights.isec_jo \
                                          * loss_isec_jo_per_transform[:-1]
                else:
                    loss_per_transform += weights.isec_jo \
                                          * loss_isec_jo_per_transform

            # TODO: keep tabs on the transform_id of each smoothness entry
            # + weights.smooth * loss_smooth_per_transform \

            optimizer_var_list = [translation_, rotation]
            if 'params_camera' in locals():
                for i, w_cam_param in enumerate(w_camera):
                    if w_cam_param:
                        lg.debug("Using camera parameter %d" % i)

            fetches_cb = None  # [transforms]
            the_callback = None  # cb_indep

            if 'maxiter' in lbfgs_options:
                lbfgs_options['maxiter'] = ceil(lbfgs_options['maxiter']
                                                * 0.666)
            optimizer = ScipyOptimizerInterface(
              loss=loss, var_list=optimizer_var_list,
              options=lbfgs_options,
              # var_to_bounds=bounds,
              method='L-BFGS-B'
            )

            if with_jo_intersection or with_occlusion:
                lbfgs_options2 = {k: v for k, v in lbfgs_options.items()}
                if 'maxiter' in lbfgs_options:
                    lbfgs_options2['maxiter'] = \
                        ceil(lbfgs_options2['maxiter'] / 2.)
                optimizer_w_isec = ScipyOptimizerInterface(
                  loss=loss_2, var_list=optimizer_var_list,
                  options=lbfgs_options2,
                  method='L-BFGS-B'
                )
        else:  # consistent
            # TODO: need oo in first?

            def compose_loss2(loss, w_isec_jo, w_isec_oo):
                _loss = loss
                if with_jo_intersection:
                    _loss += w_isec_jo * loss_isec_jo_sc
                    _loss += w_isec_oo * loss_isec_oo_sc
                else:
                    lg.error("Not using intersection in opt")
                    assert False
                    
                if with_occlusion:
                    _loss += weights.occl * loss_occl_sc
                else:
                    lg.error("Not using occlusion in loss.")
                    
                return _loss
                
            # loss_2 = loss
            # if not with_jo_intersection:
            #     lg.error("not using intersection in opt")
            #     assert False
            # else:
            #     loss_2 += weights.isec_jo * loss_isec_jo_sc
            #     loss_2 += weights.isec_oo * loss_isec_oo_sc
            #
            # if not with_occlusion:
            #     lg.error("Not using occlusion in loss.")
            # else:
            #     loss_2 += weights.occl * loss_occl_sc
            loss_2 = compose_loss2(loss=loss, 
                                   w_isec_jo=weights.isec_jo,
                                   w_isec_oo=weights.isec_oo)
            loss_half = compose_loss2(loss=loss, 
                                   w_isec_jo=weights.isec_jo * 0.01,
                                   w_isec_oo=weights.isec_oo * 0.01)
            optimizer_var_list = [translation_, rotation]
            bounds = {}
            # if 'params_camera' in locals():
            #     for i, w_cam_param in enumerate(w_camera):
            #         if w_cam_param:
            #             lg.debug("Using camera parameter %d" % i)
            optimizer_var_list.append(params_camera)
            # if w_camera[0]:
            lg.warning("Setting rotation x bound")
            bounds = {params_camera: ([-np.inf, -np.inf, -np.inf],
                                      [0., np.inf, np.inf]),
                      rotation: (-np.pi/2., np.pi/2.)}

            fetches_cb = None
            the_callback = None  # loss_callback

            pre_opt_options = {
                k:v 
               for k, v in lbfgs_options.items()}
            # pre_opt_options['maxiter'] = 200
            optimizer = ScipyOptimizerInterface(
              loss=loss_half, var_list=optimizer_var_list,
              options=pre_opt_options,
              var_to_bounds=bounds,
              method='L-BFGS-B'  # change method here
            )

            optimizer_w_isec = ScipyOptimizerInterface(
              loss=loss_2, var_list=optimizer_var_list,
              options=lbfgs_options,
              var_to_bounds=bounds,
              method='L-BFGS-B'
            )

        # optimizer
        # bounds = {translation_: ([-5, -1., 0.], [5., 1., 10.])}

    lg.info("[match] Finished graph...")

    # save tf graph to tensorboard
    # log_dir = os.path.join(os.path.dirname(__file__), 'tf_summary')
    # tf.gfile.MakeDirs(log_dir)
    # writer = tf.summary.FileWriter(log_dir, graph=graph)

    # optimize
    with Timer('solve', verbose=True):
        config = tf.ConfigProto(allow_soft_placement=False)
        with tf.Session(graph=graph, config=config) as session:
            lg.info("[match] variables_initializer")
            # Init
            session.run(tf.global_variables_initializer())

            # tmp_rot_orig = rotation.eval(session)
            # tmp_rot = tmp_rot_orig.copy()
            # tmp_rot[0] = tmp_rot_orig[0] + np.pi/2
            # session.run([rotation.assign(tmp_rot)])

            # Optimize
            if with_jo_intersection:
                lg.debug("[match] minimizing 2")
                try:
                    optimizer.minimize(session=session,
                                       loss_callback=the_callback,
                                       fetches=fetches_cb)
                    optimizer_w_isec.minimize(session=session,
                                              loss_callback=the_callback,
                                              fetches=fetches_cb)
                except UnboundLocalError:
                    lg.warning("No second minimizer step")
                lg.debug("[match] finished minimizing 2")
            else:
                lg.info("[match] minimizing")
                optimizer.minimize(session=session,
                                   loss_callback=the_callback,
                                   fetches=fetches_cb)

            lg.info("[match] finished minimizing")

            # sample_rotation(loss, loss_occl_per_transform, rotation,
            #                 loss_per_transform=loss_per_transform,
            #                 session=session, n_steps=10)

            # Manually position scenelets
            # tmp = translation_.eval(session)
            # tmp[0, 2] -= 0.25 # z of first scenelet
            # tmp[1, 2] += 1.5  # z of second scenelet
            # session.run(translation_.assign(tmp))

            # untested:
            # tmp_rot = rotation.eval(session)
            # tmp[0] = np.pi
            # session.run(rotation.assign(tmp_rot))
            # optional: reoptimize from this starting point:
            # optimizer.minimize(session=session,
            #                    loss_callback=the_callback,
            #                    fetches=fetches_cb)

            # query major losses
            o_loss, o_loss_reproj, o_smooth_diffs = \
                session.run([loss, loss_reproj, smooth_diffs])

            # NOTE: loss_occl_per_transform is not complete
            if independent:
                # query per-scenelet losses and output positions of poses
                # and 3D bounding boxes
                o_loss_per_transform, o_pos_3d, o_polys_3d = \
                    session.run([loss_per_transform, pos_3d, polys_3d])

                o_loss_reproj_per_transform = loss_reproj_per_transform.eval()

                if with_occlusion:
                    with Timer('loss_occl_per_transform'):
                        o_loss_occl_per_transform = \
                            compute_occl_loss_per_transform(
                              d_per_joint_large, len_joints_to_occlude,
                              loss_occl_per_pose, loss_occl_unn,
                              n_poses_moving, n_transforms, session, um,
                              independent=independent,
                              np_transform_indices_both=np_transform_indices_both)
                        assert o_loss_per_transform.ndim == 1
                        o_loss_per_transform += \
                            o_loss_occl_per_transform[
                            :o_loss_per_transform.shape[0]]

                if with_jo_intersection:
                    o_loss_isec_jo_per_transform = \
                        loss_isec_jo_per_transform.eval()

                # how many candidates to keep of this batch
                choice_limit = min(n_cands_per_batch, len(scenes_corresps))
                # best transform ids
                chosen = sorted(
                  (transform_id_
                   for transform_id_ in range(o_loss_per_transform.shape[0])),
                  key=lambda i2: o_loss_per_transform[i2])[:choice_limit]
                # corresponding scores to transform_ids in chosen
                scores = [float(o_loss_per_transform[tid]) for tid in chosen]

                # Need these for error reporting to terminal
                if not with_jo_intersection:
                    o_loss_isec_jo = -1
                else:
                    o_loss_isec_jo = session.run(loss_isec_jo_sc)

                o_loss_isec_oo = -1

                if with_occlusion:
                    o_loss_occl = session.run(loss_occl_sc)
                else:
                    o_loss_occl = -1.
            else:  # not independent
                o_loss_isec_jo, o_loss_isec_oo = \
                    session.run([loss_isec_jo_sc, loss_isec_oo_sc])

                if with_occlusion:
                    o_loss_occl = session.run(loss_occl_sc)
                else:
                    o_loss_occl = -1.

                print_loss(o_loss_reproj, o_smooth_diffs, o_loss_isec_jo,
                           o_loss_isec_oo, o_loss, o_loss_occl, weights)

                show_output(tf_vars, deb_oo, deb_jo, d_query, session,
                            smooth_pairs, d_postfix='_auto', f_postfix='',
                            um=um)
                o_pos_3d, o_polys_3d = session.run([pos_3d, polys_3d])

                if 'params_camera' in locals():
                    lg.debug("camera is\n%s" % params_camera.eval())
                    lg.debug("camera is\n%s" % transform_camera.eval())

    # Error reporting
    print_loss(o_loss_reproj, o_smooth_diffs, o_loss_isec_jo,
               o_loss_isec_oo, o_loss, o_loss_occl, weights)

    #
    # Save outputs
    #

    with Timer('save outputs'):
        out_scenelets = {}  # {candidate_rank in batch: scenelet, ...}
        # rate = query_full.skeleton.get_rate()

        def add_scores(sclt_arg, transform_id_arg):
            sclt_arg.add_aux_info(
              'score_reproj',
              float(o_loss_reproj_per_transform[transform_id_arg]))
            sclt_arg.add_aux_info('weight_score_reproj', float(
              weights.proj))
            sclt_arg.add_aux_info('weight_score_smooth',
                                  float(weights.smooth))
            if with_occlusion:
                sclt_arg.add_aux_info(
                  'score_occl',
                  float(o_loss_occl_per_transform[transform_id_arg]))
                sclt_arg.add_aux_info('weight_score_occl', float(
                  weights.occl))
            if with_jo_intersection:
                sclt_arg.add_aux_info(
                  'score_isec_jo',
                  float(o_loss_isec_jo_per_transform[transform_id_arg]))
                sclt_arg.add_aux_info('weight_score_isec_jo',
                                      float(weights.isec_jo))
                sclt_arg.add_aux_info('weight_score_isec_oo',
                                      float(weights.isec_oo))

            return sclt_arg

        if independent:
            lg.info("Best candidate is %d with error %g"
                    % (chosen[0], scores[0]))
            # TODO: iterate over chosen until all scenes have one candidate
            candidates = list(zip(chosen, scores))
            for cand_id, (transform_id, score) in enumerate(candidates):
                sclt = export_scenelet(um=um, o_pos_3d=o_pos_3d,
                                       o_polys_3d=o_polys_3d,
                                       query_full_skeleton=query_full.skeleton,
                                       scenes=scenes,
                                       joints_active=joints_active,
                                       transform_id=transform_id)
                sclt.score_fit = float(score)
                add_scores(sclt, transform_id)
                out_scenelets[cand_id] = sclt
        else:  # not independent
            o = export_scenelet(um=um, o_pos_3d=o_pos_3d,
                                o_polys_3d=o_polys_3d,
                                query_full_skeleton=query_full.skeleton,
                                scenes=scenes,
                                joints_active=joints_active)
            # add_scores(o, transform_id)
            out_scenelets[0] = o

    return out_scenelets


def read_fill_scenes(d_scenelets, dist_thresh):
    """Reads scenelets that have been aligned in time already.
    Usually called for the second optimization.
    :param d_scenelets:
    :param dist_thresh:
    :return:
    """
    p_scenelets_pickle = os.path.join(d_scenelets, 'candidates_opt3.pickle')
    if False and os.path.exists(p_scenelets_pickle):
        scenes = pickle_load(open(p_scenelets_pickle, 'rb'))
    else:
        scenes = read_scenelets(d_scenelets, limit=0,
                                dist_thresh=dist_thresh)
        pickle.dump(scenes, open(p_scenelets_pickle, 'wb'))
    return scenes


def read_gap_scenes(d_scenelets, no_pickle=False, filter_fn=None):
    """
    Args:
        filter_fn (function):
            Returns true, if we want to keep the scenelet. Takes scenelet
            name as argument.
    """
    is_pickled = False
    p_scenelets_pickle = os.path.join(d_scenelets, 'scenes_db.pickle')
    if not no_pickle and os.path.exists(p_scenelets_pickle):
        lg.warning("Reading from %s" % p_scenelets_pickle)
        scenes = pickle_load(open(p_scenelets_pickle, 'rb'))
        lg.warning("Read from %s" % p_scenelets_pickle)
        is_pickled = True
    else:
        scenes = read_orig_scenelets(d_scenelets, limit=0)
        if len(scenes) <= 0:
            raise RuntimeError("No scenes in %s" % d_scenelets)

    print('scenes: %s' % len(scenes))
    blacklist = {} #'sga15-gates200Hallway_1_2015-05-18-22-06-27__scenelet_2'}
    if len(blacklist):
        if filter_fn is not None:
            scenes_filtered = [s for s in scenes
                               if s.name_scenelet not in blacklist
                               and filter_fn(s.name_scenelet)]
        else:
            scenes_filtered = [s for s in scenes
                               if s.name_scenelet not in blacklist]
    else:
        if filter_fn is not None:
            scenes_filtered = [s for s in scenes
                               if filter_fn(s.name_scenelet)]
        else:
            scenes_filtered = scenes

    if len(scenes) > 0:
        if not no_pickle \
          and (not is_pickled or len(scenes_filtered) != len(scenes)):
            pickle.dump(scenes, open(p_scenelets_pickle, 'wb'))

    return scenes_filtered


def split_seq(iterable, size):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def filter_same_scene(name_scenelet, scene_names):
    assert os.sep not in name_scenelet, "No %s" % name_scenelet
    name_scene = name_scenelet.partition('_')[0]
    ret = name_scene not in scene_names
    if ret:
        print("keeping %s" % name_scenelet)
    else:
        print("throwing %s" % name_scenelet)

    return ret


def main(argv, cache_scenes=None):
    np.set_printoptions(suppress=True, linewidth=200)
    pjoin = os.path.join

    parser = argparse.ArgumentParser("matcher")
    subparsers = parser.add_subparsers(
      title='variants',
      description='You can either optimize consistently or independently')

    #
    # Common
    #

    parser.add_argument("-v", "--video", type=argparse_check_exists,
                        help="Input path")
    parser.add_argument(
        "--dist_thresh", type=float,
        help="Scenelet to skeleton distance threshold that decides, "
             "whether an object should be kept for optimization or not.",
        default=.15
    )
    parser.add_argument(
        '--wp', type=float, help="Projection weight.", default=1.
    )
    parser.add_argument(
        '--ws', type=float, help="Smoothness weight.", default=.1
    )
    parser.add_argument(
      '--wo', type=float, help="Occlusion weight (remember to multiply by "
                               "~10^2 because of the distance being in m).",
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
    parser.add_argument('--with-y', action='store_true',
                        help="Optimize for y translation")
    parser.add_argument('-w-occlusion', action='store_true',
                        help="Estimate occlusion score.")
    parser.add_argument('-w-static-occlusion', action='store_true',
                        help="Occlude already placed paths.")
    parser.add_argument('-silent', action='store_true',
                        help="Print less to console")
    parser.add_argument('-no-isec', action='store_true',
                        help='Don\'t use intersection terms')
    parser.add_argument('--output-n', type=int,
                        help="How many candidates to output per batch and "
                             "overall.", default=200)
    parser.add_argument('--d-pad', type=float,
                        help="Add space around objects for joint-object "
                             "intersection to account for body size.",
                        default=0.)

    parser.add_argument('--w-camera', action='store_true',
                        help="Allow camera")

    parser.add_argument('--nomocap', action='store_true',
                        help="Don't use mocap scenelets")
    parser.add_argument('--filter-scenes', nargs='*', type=str,
                        help='Hold out scenelets from these scenes')

    #
    # Consistent
    #

    parser_c = subparsers.add_parser(
      'consistent',
      help='Globally consistent optimization with intersection, spatial '
           'smoothness and moving local poses.')
    parser_c.add_argument('consistent', action='store_const', const=True)
    parser_c.add_argument('independent', action='store_const', const=False)
    parser_c.add_argument('-input', type=str, help="Input scenelet folder",
                          required=True)
    parser_c.add_argument('--no-legs', action='store_true')

    #
    # Independent
    #

    parser_i = subparsers.add_parser(
      'independent',
      help='Independent search for explanations for a part of a video.')
    parser_i.add_argument('independent', action='store_const', const=True)
    parser_i.add_argument('consistent', action='store_const', const=False)
    parser_i.add_argument("-s", "--d-scenelets", type=argparse_check_exists,
                          help="Folder containing original PiGraphs scenelets")
    parser_i.add_argument("--gap-size-limit", type=int,
                          help="Smallest gap size to still explain",
                          default=12)
    parser_i.add_argument('-tc', '--thresh-charness', type=float,
                          help="Input scenelets lower charness threshold.",
                          required=True)
                          #default=0.35)
    parser_i.add_argument('--gap', type=int, nargs=2,
                          help="Specify start and stop frame_id (inclusive) "
                               "to fit to")
    parser_i.add_argument('--candidates', type=str,
                          help="Folder containing already prefit scenelets")
    parser_i.add_argument('--n-candidates', type=int,
                          help="How many candidates to consider")
    parser_i.add_argument('--dest-dir', type=str,
                          help="Name of subdirectory to save output to.",
                          default='opt1')
    parser_i.add_argument('--batch-size', type=int,
                          help="How many scenelets to optimize at once.",
                          default=3000)
    parser_i.add_argument('--candidate', type=str, nargs='*',
                          help="Path to a single scenelet json file.")
    parser_i.add_argument('--out-limit', type=int,
                          help="How many scenelets to output to root.",
                          default=9)
    parser_i.add_argument('--occlusion-eval-only', action='store_true',
                          help="Only evaluate occlusion, don't optimize",
                          default=True)
    parser_i.add_argument('--remove-objects', action='store_true',
                          help="Do not output objects, just poses",
                          default=False)
    # parser_i.add_argument('-tc', '--thresh-charness', type=float,
    #                       help="Lower threshold to charness of scenelets. "
    #                            "Default: 0.35",
    #                       required=True)

    # parse
    args = parser.parse_args(argv)
    if 'candidates' in args and args.candidates and not args.n_candidates:
        lg.error("Need how many candidates! %s %s"
                 % (args.candidates, args.n_candidates))
        sys.exit(0)
    cats_to_ignore = frozenset(('book', 'wall', 'floor', 'whiteboard',
                                'notebook', 'laptop', 'plant', 'monitor'))
    if 'filter_scenes' in args and args.filter_scenes \
      and len(args.filter_scenes):
        sclt_filter_fn = partial(filter_same_scene,
                                 scene_names=args.filter_scenes)
    else:
        sclt_filter_fn = None

    if '--gtol' not in argv and args.consistent:
        args.gtol = 1.e-12
        lg.warning("Overwriting gtol to %g" % args.gtol)

    assert args.output_n >199

    # save call log to source folder
    d_src = os.path.dirname(__file__)
    with open(pjoin(d_src, 'args.txt'), 'a') as f_args:
        f_args.write("(cd %s) && " % os.getcwd())
        f_args.write('(python3 ')
        f_args.write(" ".join(sys.argv))
        f_args.write(")\n")

    # get video parent directory
    d_query = args.video if os.path.isdir(args.video) \
        else os.path.dirname(args.video)

    # save call log to video directory
    with open(pjoin(d_query, 'args_opt_consistent.txt'), 'a') as f_args:
        f_args.write('(python3 ')
        f_args.write(" ".join(sys.argv))
        f_args.write(")\n")

    # parse weights
    weights = Weights(proj=args.wp, smooth=args.ws,
                      isec_oo=args.wi if 'wi' in args else 0.,
                      occl=args.wo)
    # optimization parameters
    opt_params = OptParams(args.gtol, args.maxiter)

    # parse video path
    name_query = os.path.split(d_query)[-1]
    assert len(name_query), "No: %s" % name_query
    p_query = pjoin(d_query, "skel_%s_unannot.json" % name_query) \
        if os.path.isdir(args.video) else args.video
    assert p_query.endswith('.json'), "Need a skeleton file"
    # print("here: %s" % p_query)
    # sys.exit(0)

    if not os.path.exists(p_query):
        p_query = pjoin(d_query, "skel_%s.json" % name_query)

    # load initial video path (local poses)
    query = Scenelet.load(p_query, no_obj=True)
    tr_ground = np.array(query.aux_info['ground'], dtype=np.float32)
    if np.dot(tr_ground[:3, 1], [0, -1., 0.]) < 0.:
        lg.error("Flipping ground transform: %s" % tr_ground)
        tr_ground[:, 1] *= -1.
    if not args.silent:
        lg.debug("ground transform:\n%s" % tr_ground)
    assert 'ground_rot' in query.aux_info, "need ground rotation"
    ground_rot = query.aux_info['ground_rot']
    ground_rot[0] = np.deg2rad(ground_rot[0])

    # load 2D keypoints
    query_2d_skeleton = Scenelet.load(
      pjoin(d_query, "skel_%s_2d_00.json" % name_query)).skeleton

    # load intrinsics
    p_intr = pjoin(d_query, 'intrinsics.json')
    intr = np.array(json.load(open(p_intr, 'r')), dtype=np.float32)
    if not args.silent:
        lg.debug("intrinsics:\n%s" % intr)

    # load scenes to fit
    scenes = []
    scenes_transformed = None
    if args.independent:
        candidate_names = []
        if args.candidate is not None:
            # compose path
            for p_cand in args.candidate:
                p_candidate = pjoin(d_query, p_cand)
                # load scenelet
                scenes.append(Scenelet.load(p_candidate))
            # where to save output
            args.dest_dir = "%s_candidate" % args.dest_dir
            # this scenelet is already in room space
            scenes_transformed = False
        elif 'candidates' in args and args.candidates is not None:
            d_candidates = pjoin(d_query, args.candidates)
            assert os.path.exists(d_candidates), d_candidates
            assert os.path.isdir(d_candidates), d_candidates
            p_sclts = []
            for p_sclt in listdir(d_candidates):
                fname = os.path.basename(p_sclt)
                if not fname.endswith('json') or not fname.startswith('skel_'):
                    if os.path.isdir(p_sclt) and '_more' in p_sclt:
                        for _p_sclt in listdir(p_sclt):
                            if _p_sclt.endswith('json') and \
                              os.path.basename(_p_sclt).startswith('skel_'):
                                p_sclts.append(_p_sclt)
                else:
                    p_sclts.append(p_sclt)

            for p_sclt in p_sclts:
                _sclt = Scenelet.load(p_sclt, no_obj=True)
                candidate_names.append(os.path.basename(_sclt.name_scenelet))
                if 'n_candidates' in args \
                  and len(candidate_names) >= args.n_candidates:
                    break
            assert len(candidate_names), "No candidates!"
            args.dest_dir = "%sb" % args.dest_dir

        if not len(scenes):
            if cache_scenes is not None:
                scenes = copy.deepcopy(cache_scenes)
                assert len(scenes), "Cache empty??"
            else:
                assert os.path.exists(args.d_scenelets), "Need scenelets"
                with Timer('read scenelets'):
                    scenes = read_gap_scenes(args.d_scenelets,
                                             filter_fn=sclt_filter_fn)
                    assert len(scenes), "just read...No scenes?"
            # these scenelets are not in room space
            scenes_transformed = False

        if 'gap' not in args:
            gaps = find_gaps(query.skeleton, min_pad=1)
            gaps = [gap for gap in gaps
                    if gap[1] - gap[0] >= args.gap_size_limit]
            gaps = [gaps[1]]
        else:
            gaps = [args.gap]
            if len(candidate_names):
                lg.debug("filtering: %s" % candidate_names)
                scenes = [scene for scene in scenes
                          if os.path.basename(scene.name_scenelet)
                          in candidate_names]
        lg.debug("gaps: %s" % gaps)
    else:
        assert args.consistent, "args.consistent: %s" % args
        d_scenelets = pjoin(d_query, args.input)
        assert os.path.exists(d_scenelets), "no: %s" % d_scenelets
        scenes = read_gap_scenes(d_scenelets, no_pickle=True)
        # gaps = [scn.skeleton.get_frames_min_max() for scn in scenes]
        gaps = None
        correspondences = [
            [(frame_id, frame_id)
             for frame_id in scenelet.skeleton.get_frames()]
            for scenelet in scenes]

        # these scenelets are already in room space
        scenes_transformed = True

    if isinstance(scenes, bool) and not scenes or not len(scenes):
        lg.error("NO SCENES")
        return [] if cache_scenes is None else cache_scenes

    # assert len(scenes), "No scenes?"

    # work
    with Timer("outer", verbose=True):
        if args.independent:
            fps = 3
            assert len(gaps) == 1, "todo"
            gap = gaps[0]

            if not args.silent:
                plot_charness(scenes, d_dir=args.d_scenelets)

            if args.thresh_charness > 0.:
                lg.warning("Filtering charness at %g" % args.thresh_charness)
            scenes_corresps = sorted(
                [
                    (
                        sclt,
                        UnkManager.compute_correspondence(
                            query_start=gap[0], query_end=gap[1],
                            frame_ids_scene=sclt.skeleton.get_frames(),
                            fps=fps, stretch=False, clip=True)
                    )
                    for sclt in scenes
                    if sclt.charness > args.thresh_charness
                       and (
                               not args.nomocap
                               or ('take' not in sclt.name_scenelet)
                       )
                ],
                key=lambda s_c: s_c[0].charness,
                reverse=True
            )
            lg.info("Have %d scenelets left" % len(scenes_corresps))
            bests = {}
            chunk_start = 0
            batch_id = 0
            n_batches = int(ceil(len(scenes_corresps)
                                 / float(args.batch_size)))
            for scene_chunk in split_seq(scenes_corresps, args.batch_size):
                lg.info("[main] batch %d/%d" % (batch_id + 1, n_batches))
                with Timer('batch') as t_batch:
                    out_sclts = match(
                      query_full=query,
                      d_query=d_query,
                      query_2d_full=query_2d_skeleton,
                      scenes_corresps=scene_chunk,
                      intrinsics=intr,
                      tr_ground=tr_ground,
                      weights=weights,
                      with_occlusion=args.w_occlusion,
                      occlusion_eval_only=args.occlusion_eval_only,
                      is_scenes_transformed=scenes_transformed,
                      opt_params=opt_params,
                      z_only=not args.independent,
                      independent=args.independent,
                      with_jo_intersection=not args.no_isec,
                      with_y=args.with_y,
                      # w_camera=not args.independent or args.w_camera,
                      w_camera=(True, False, False),
                      camera_init=(ground_rot[0], 0., -0.),
                      w_static_occlusion=args.w_static_occlusion,
                      cats_to_ignore=cats_to_ignore,
                      silent=args.silent,
                      n_cands_per_batch=args.output_n,
                      min_poses=1 if args.remove_objects else 10)
                    batch_id += 1
                    if out_sclts is False:
                        continue
                    for k, sclt in out_sclts.items():
                        assert chunk_start + k not in bests
                        sclt.aux_info['score_fit'] = float(sclt.score_fit)
                        bests[chunk_start + k] = sclt
                    # lg.debug("stored occlusion score %s" % bests)
                    chunk_start += len(scene_chunk)
                with open(pjoin(d_query, 'time_stats.txt'), 'a') \
                        as f_timestats:
                    f_timestats.write(
                      "{secs:f},{maxiter:d},{gtol:f},"
                      "{batchsize:f},{radius:s},{weightoccl:f}\n".format(
                        secs=float(t_batch.secs),
                        maxiter=int(opt_params.maxiter),
                        gtol=float(opt_params.gtol),
                        batchsize=int(args.batch_size),
                        radius=str(os.path.split(args.d_scenelets)[-1]),
                        weightoccl=float(weights.occl)))
            out_sclts = bests
        else:  # consistent
            assert len(scenes) == len(correspondences)
            scenes_corresps = list(zip(scenes, correspondences))
            out_sclts = match(
              query_full=query,
              d_query=d_query,
              query_2d_full=query_2d_skeleton,
              scenes_corresps=scenes_corresps,
              intrinsics=intr,
              tr_ground=tr_ground,
              weights=weights,
              with_occlusion=args.w_occlusion,
              is_scenes_transformed=scenes_transformed,
              opt_params=opt_params,
              z_only=False,
              with_y=False,  # TODO: uncomment
              independent=args.independent,
              with_jo_intersection=not args.no_isec,
              w_camera=(True, True, True),  # not args.independent,
              camera_init=(ground_rot[0], 0., 0.),  # rx, ry, ty
              w_static_occlusion=args.w_static_occlusion,
              d_padding=args.d_pad,
              cats_to_ignore=cats_to_ignore,
              silent=args.silent,
              n_cands_per_batch=args.output_n,
              no_legs=args.no_legs,
              min_poses=1)
    if not len(out_sclts):
        return [] if cache_scenes is None else cache_scenes

    # save
    if args.consistent:
        if 'dest_dir' in args and args.dest_dir != 'opt1':
            p_out = pjoin(d_query, args.dest_dir + "_deb")
        else:
            p_out = pjoin(d_query, 'output')
        makedirs_backed(p_out)
        for id_scene, sclt in out_sclts.items():
            postfix = "_%02d" if len(out_sclts) > 1 else ""
            name_sclt = "skel_output%s.json" % postfix
            p_sclt = pjoin(p_out, name_sclt)
            sclt.save(p_sclt)
        d_scenelets = pjoin(d_query, args.input)

        # save candidates:
        p_out_candidates = os.path.join(p_out, 'candidates')
        if not os.path.isdir(p_out_candidates):
            os.makedirs(p_out_candidates)
        shutil.copytree(
          d_scenelets,
          os.path.join(p_out_candidates, os.path.basename(args.input)))

        with open(pjoin(d_query, 'args_consistent.txt'), 'a') as fout:
            fout.write(" ".join("%s" % av for av in sys.argv))
            fout.write("\n")
        with open(pjoin(p_out, 'params.txt'), 'w') as fout:
            fout.write(" ".join("%s" % av for av in sys.argv))
            fout.write("\n")
    else:
        max_fit = max(0.3, max(sclt.score_fit for sclt in out_sclts.values()))
        if max_fit > 0.3:
            lg.warning("Exceeded max fit: %g" % max_fit)
        weights = [(max_fit - sclt.score_fit) / max_fit
                   for sclt in out_sclts.values()]
        sum_score_fit = sum(weights)
        if abs(sum_score_fit) < 1e-3:
            sum_score_fit = 1.
        avg_charness = sum(weight * sclt.charness
                           for sclt, weight
                           in zip(out_sclts.values(), weights)) / sum_score_fit
        max_charness = max(sclt.charness for sclt in out_sclts.values())

        p_out = pjoin(d_query, args.dest_dir, "%03d_%03d"
                      % (args.gap[0], args.gap[1]))
        makedirs_backed(p_out)
        # if os.path.exists(p_out):
        #     shutil.rmtree(p_out)
        # os.makedirs(p_out)
        with open(pjoin(p_out, 'avg_charness.json'), 'w') as f_out:
            json.dump({'avg_charness': avg_charness,
                       'max_charness': max_charness}, f_out)
        # sort bests
        out_sclts_sorted = sorted(
          ((key, out_sclt) for key, out_sclt in out_sclts.items()),
          key=lambda e: e[1].score_fit)
        for sid, (key, out_sclt) in enumerate(out_sclts_sorted):
            lg.info("Saving scenelet with score %g, charness: %g"
                    % (out_sclt.score_fit, out_sclt.charness))
            mn_frame_id, mx_frame_id = out_sclt.skeleton.get_frames_min_max()
            if args.remove_objects:
                out_sclt.objects = {}

            if sid > args.out_limit:
                if out_sclt.charness > 0.2:
                    p_out_more = pjoin(p_out, '_more')
                    if not os.path.exists(p_out_more):
                        os.makedirs(p_out_more)
                    p_sclt = pjoin(p_out_more,
                                   "skel_%s_%02d.json" % (name_query, sid))
                    out_sclt.save(p_sclt)
            else:
                p_sclt = pjoin(p_out,
                               "skel_%s_%02d.json" % (name_query, sid))
                out_sclt.save(p_sclt)

    lg.info("matched %d scenelets" % len(scenes_corresps))
    # plt.show()

    if cache_scenes is not None:
        return cache_scenes
    else:
        return scenes


if '__main__' == __name__:
    # setproctitle.setproctitle("[Aron] opt_consistent.py")
    # pydevd.settrace('thorin.cs.ucl.ac.uk', port=63342, stdoutToServer=True, stderrToServer=True)
    main(sys.argv[1:])
