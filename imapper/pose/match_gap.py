import argparse
import copy
import os
import shutil
import sys
from math import ceil, floor

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

from imapper.logic.scenelet import Scenelet
from imapper.pose.config import INPUT_SIZE
from imapper.util.my_argparse import argparse_check_exists
from imapper.util.json import json
from imapper.util.my_pickle import pickle, pickle_load
from imapper.util.stealth_logging import lg
from imapper.util.timer import Timer


def rlencode(x, dropna=False):
    """Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Source: https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66


    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return list(zip(starts, lengths, values))


def find_gaps(skel, min_pad=3):
    """

    :param skel:
        Input skeleton
    :param min_pad:
        Minimum number of existing frames to have.
    :return:
        list of frame_ids, [(start, end), ..] that start with min_pad
        number of existing frames, and end with min_pad number of existing
        frames, whilst having missing frames inbetween.
    """
    frame_ids = skel.get_frames()
    indicator = np.zeros((frame_ids[-1]+1), dtype=int)
    indicator[frame_ids] = 1
    entries = rlencode(indicator)
    out = []
    for i, (s, l, v) in enumerate(entries):
        if v == 1:
            continue
        assert entries[i-1][2] == 1, "previous is also 0?"
        assert entries[i+1][2] == 1, "next is also 0?"
        if entries[i-1][1] < min_pad:
            lg.warning("Todo: extend beyond first neighbours: %s"
                       % entries[i-3:i])
            continue
        if entries[i+1][1] < min_pad:
            lg.warning("Todo: extend beyond first neighbours: %s"
                       % entries[i:i+4])
            continue
        out.append((entries[i-1][0] + entries[i-1][1] - min_pad,
                    entries[i+1][0] + min_pad - 1))
        # lg.debug("gap: %s" % repr((s, l, v)))
    out = sorted(out, key=lambda g: g[1] - g[0], reverse=True)
    return out


def read_scenelets(d_scenelets, limit=0):
    pjoin = os.path.join

    # get full pigraph scenes
    p_scenelets = [pjoin(d_scenelets, d)
                   for d in os.listdir(d_scenelets)
                   if os.path.isdir(pjoin(d_scenelets, d))]
    # for
    p_scenelets = [pjoin(d, f)
                   for d in p_scenelets
                   for f in os.listdir(d)
                   if f.startswith('skel') and f.endswith('.json')]

    out = []
    for p_scenelet in p_scenelets:
        lg.info("Reading %s" % p_scenelet)
        out.append(Scenelet.load(p_scenelet))
        if limit != 0 and len(out) >= limit:
            break

    return out


def get_partial_scenelet(scene, start=None, end=None,
                         mid_frame_id=None, n_frames=None, fps=3):
    if start is None or end is None:
        half = n_frames // 2
        start = int(floor(mid_frame_id - half * fps))
        end = int(ceil(mid_frame_id + half * fps))
        if (end - start) / fps != n_frames:
            end = int(ceil(mid_frame_id + (half+1) * fps))
        assert (end - start) / fps == n_frames, \
            "no: %s (%d -%d) vs %s" % ((end-start), end, start, n_frames)
    else:
        assert start is not None and end is not None
        n_frames = end - start

    # lg.debug("start, end: %d, %d" % (start, end))
    frame_ids = scene.skeleton.get_frames()
    if start < frame_ids[0]:
        lg.error("partial scene outside scene: %s < %s"
                 % (start, frame_ids[0]))
        return False
    elif end > frame_ids[-1]:
        lg.error("partial scene outside scene: %s > %s"
                 % (end, frame_ids[-1]))
        return False

    shp = scene.skeleton.poses.shape
    poses = np.zeros((n_frames, shp[1], shp[2]), dtype=np.float32)
    indicator = np.zeros((n_frames,), dtype=np.int32)
    for lin_id, frame_id in enumerate(range(start, end, int(fps))):
        try:
            poses[lin_id, :, :] = scene.skeleton.get_pose(frame_id)
            indicator[lin_id] = 1
        except KeyError:
            pass
    assert lin_id == n_frames - 1, "%d %d" % (lin_id, n_frames)

    return poses, indicator


def match(query_full, d_query, query_2d_full, scene, intr, gap, tr_ground,
          scale, thresh_log_conf=7.5, w_3d=0.01, fps=3, step_samples=100):
    with_y = False  # optimize for y as well
    np.set_printoptions(suppress=True, linewidth=220)

    pjoin = os.path.join

    len_gap = gap[1] - gap[0] + 1
    query, q_v = get_partial_scenelet(query_full, start=gap[0], end=gap[1]+1,
                                      fps=1)
    q_v_sum = np.sum(q_v)
    q_v_sum_inv = np.float32(1./q_v_sum)
    # lg.debug("q_v_sum: %s/%s" % (q_v_sum, q_v.size))
    # scene_min_y = scene.skeleton.get_min_y(tr_ground)
    # lg.debug("scene_min_y: %s" % repr(scene_min_y))

    mid_frames = range(len_gap*fps,
                       scene.skeleton.poses.shape[0]-len_gap*fps,
                       step_samples)
    if not len(mid_frames):
        return []

    scenelets, sc_v = (np.array(e) for e in zip(
        *[
            get_partial_scenelet(scene, mid_frame_id=mid_frame_id,
                                 n_frames=len_gap, fps=fps)
            for mid_frame_id in mid_frames
        ]
    ))
    # for i, (scenelet, sc_v_) in enumerate(zip(scenelets, sc_v)):
    #     mn = np.min(scenelet[sc_v_.astype('b1'), 1, :])
    #     scenelets[i, :, 1, :] -= mn
        # mn = np.min(scenelets[i, sc_v_.astype('b1'), 1, :])
    # scenelets = np.array(scenelets, dtype=np.float32)
    # sc_v = np.array(sc_v, dtype=np.int32)
    # print("sc_v: %s" % sc_v)
    # print("q_v: %s" % q_v)

    lg.debug("have %d/%d 3D poses in scenelet, and %d/%d in query"
             % (np.sum(sc_v), sc_v.shape[0], np.sum(q_v), q_v.shape[0]))

    query_2d = np.zeros((len_gap, 2, 16), dtype=np.float32)
    conf_2d = np.zeros((len_gap, 1, 16), dtype=np.float32)
    for lin_id, frame_id in enumerate(range(gap[0], gap[1]+1)):

        if query_2d_full.has_pose(frame_id):
            query_2d[lin_id, :, :] = query_2d_full.get_pose(frame_id)[:2, :]
        # else:
        #     lg.warning("Query2d_full does not have pose at %d?" % frame_id)

        # im = im_.copy()
        if query_2d_full.has_confidence(frame_id):
            # print("showing %s" % frame_id)
            for joint, conf in query_2d_full._confidence[frame_id].items():
                log_conf = abs(np.log(conf)) if conf >= 0. else 0.
                # print("conf: %g, log_conf: %g" % (conf, log_conf))
                # if log_conf <= thresh_log_conf:
                #     p2d = scale * query_2d_full.get_joint_3d(joint,
                #                                              frame_id=frame_id)
                #     p2d = (int(round(p2d[0])), int(round(p2d[1])))
                #     cv2.circle(im, center=p2d,
                #                radius=int(round(3)),
                #                color=(1., 1., 1., 0.5), thickness=1)
                conf_2d[lin_id, 0, joint] = max(
                    0.,
                    (thresh_log_conf - log_conf) / thresh_log_conf
                )

            # cv2.imshow('im', im)
            # cv2.waitKey(100)
    # while cv2.waitKey() != 27: pass
    conf_2d /= np.max(conf_2d)

    # scale from Denis' scale to current image size
    query_2d *= scale

    # move to normalized camera coordinates
    query_2d -= intr[:2, 2:3]
    query_2d[:, 0, :] /= intr[0, 0]
    query_2d[:, 1, :] /= intr[1, 1]

    #
    # initialize translation
    #

    # centroid of query poses
    c3d = np.mean(query[q_v.astype('b1'), :, :], axis=(0, 2))
    # estimate scenelet centroids
    sclt_means = np.array([
        np.mean(scenelets[i, sc_v[i, ...].astype('b1'), ...], axis=(0, 2))
        for i in range(scenelets.shape[0])
    ], dtype=np.float32)
    # don't change height
    sclt_means[:, 1] = 0
    scenelets -= sclt_means[:, None, :, None]
    lg.debug("means: %s" % repr(sclt_means.shape))
    if with_y:
        np_translation = np.array([c3d
                                   for i in range(scenelets.shape[0])],
                                  dtype=np.float32)
    else:
        np_translation = np.array([c3d[[0, 2]]
                                   for i in range(scenelets.shape[0])],
                                  dtype=np.float32)
    np_rotation = np.array([np.pi * (i % 2)
                            for i in range(scenelets.shape[0])],
                           dtype=np.float32)[:, None]
    n_cands = np_translation.shape[0]
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        # 3D translation
        translation_ = tf.Variable(initial_value=np_translation,
                                   name='translation', dtype=tf.float32)
        t_y = tf.fill(dims=(n_cands,),
                      value=(tr_ground[1, 3]).astype(np.float32))
        # t_y = tf.fill(dims=(n_cands,), value=np.float32(0.))
        lg.debug("t_y: %s" % t_y)
        if with_y:
            translation = translation_
        else:
            translation = tf.concat(
                (translation_[:, 0:1],
                 t_y[:, None],
                 translation_[:, 1:2]),
                axis=1
            )

        lg.debug("translation: %s" % translation)
        # 3D rotation (Euler XYZ)
        rotation = tf.Variable(np_rotation, name='rotation', dtype=tf.float32)
        # lg.debug("rotation: %s" % rotation)

        w = tf.Variable(conf_2d, trainable=False, name='w', dtype=tf.float32)

        pos_3d_in = tf.Variable(query,
                                trainable=False, name='pos_3d_in',
                                dtype=tf.float32)
        # pos_3d_in = tf.constant(query, name='pos_3d_in', dtype=tf.float32)

        pos_2d_in = tf.Variable(query_2d,
                                trainable=False, name='pos_2d_in',
                                dtype=tf.float32)
        # pos_2d_in = tf.constant(query_2d, name='pos_2d_in',
        #                         dtype=tf.float32)

        pos_3d_sclt = tf.Variable(scenelets,
                                  trainable=False, name='pos_3d_sclt',
                                  dtype=tf.float32)
        # print("pos_3d_sclt: %s" % pos_3d_sclt)

        # rotation around y
        my_zeros = tf.zeros((n_cands, 1), dtype=tf.float32, name='my_zeros')
        # tf.add_to_collection('to_init', my_zeros)
        my_ones = tf.ones((n_cands, 1))
        # tf.add_to_collection('to_init', my_ones)
        c = tf.cos(rotation, 'cos')
        # tf.add_to_collection('to_init', c)
        s = tf.sin(rotation, 'sin')
        # t0 = tf.concat([c, my_zeros, -s], axis=1)
        # t1 = tf.concat([my_zeros, my_ones, my_zeros], axis=1)
        # t2 = tf.concat([s, my_zeros, c], axis=1)
        # transform = tf.stack([t0, t1, t2], axis=2, name="transform")
        # print("t: %s" % transform)
        transform = tf.concat(
            [c, my_zeros, -s,
             my_zeros, my_ones, my_zeros,
             s, my_zeros, c], axis=1)
        transform = tf.reshape(transform, ((-1, 3, 3)), name='transform')
        print("t2: %s" % transform)
        # lg.debug("transform: %s" % transform)

        # transform to 3d
        # pos_3d = tf.matmul(transform, pos_3d_sclt) \
        #          + tf.tile(tf.expand_dims(translation, 2),
        #                    [1, 1, int(pos_3d_in.shape[2])])
        # pos_3d = tf.einsum("bjk,bcjd->bcjd", transform, pos_3d_sclt)
        shp = pos_3d_sclt.get_shape().as_list()
        transform_tiled = tf.tile(transform[:, None, :, :, None],
                                  (1, shp[1], 1, 1, shp[3]))
        # print("transform_tiled: %s" % transform_tiled)
        pos_3d = tf.einsum("abijd,abjd->abid", transform_tiled, pos_3d_sclt)
        # print("pos_3d: %s" % pos_3d)
        pos_3d += translation[:, None, :, None]
        #pos_3d = pos_3d_sclt
        # print("pos_3d: %s" % pos_3d)

        # perspective divide
        # pos_2d = tf.divide(
        #     tf.slice(pos_3d, [0, 0, 0], [n_cands, 2, -1]),
        #     tf.slice(pos_3d, [0, 2, 0], [n_cands, 1, -1]))
        pos_2d = tf.divide(pos_3d[:, :, :2, :],
                           pos_3d[:, :, 2:3, :])

        # print("pos_2d: %s" % pos_2d)

        diff = pos_2d - pos_2d_in
        # mask loss by 2d key-point visibility
        # print("w: %s" % w)
        # w_sum = tf.reduce_sum()
        masked = tf.multiply(diff, w)
        # print(masked)
        # loss_reproj = tf.nn.l2_loss(masked)
        # loss_reproj = tf.reduce_sum(tf.square(masked[:, :, 0, :])
        #                             + tf.square(masked[:, :, 1, :]),
        #                             axis=[1, 2])
        masked_sqr = tf.square(masked[:, :, 0, :]) \
                     + tf.square(masked[:, :, 1, :])
        loss_reproj = tf.reduce_sum(masked_sqr, axis=[1, 2])
        # lg.debug("loss_reproj: %s" % loss_reproj)

        # distance from existing 3D skeletons
        d_3d = q_v_sum_inv * tf.multiply(pos_3d - query[None, ...],
                                         q_v[None, :, None, None],
                                         name='diff_3d')
        # print(d_3d)

        loss_3d = w_3d * tf.reduce_sum(tf.square(d_3d[:, :, 0, :])
                                       + tf.square(d_3d[:, :, 1, :])
                                       + tf.square(d_3d[:, :, 2, :]),
                                       axis=[1, 2], name='loss_3d_each')
        # print(loss_3d)

        loss = tf.reduce_sum(loss_reproj) + tf.reduce_sum(loss_3d)

        # optimize
        optimizer = ScipyOptimizerInterface(
            loss, var_list=[translation_, rotation],
            options={'gtol': 1e-12}
        )

    with Timer('solve', verbose=True) as t:
        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            optimizer.minimize(session)
            o_pos_3d, o_pos_2d, o_masked, o_t, o_r, o_w, o_d_3d, \
                o_loss_reproj, o_loss_3d, o_transform, o_translation = \
                session.run([
                    pos_3d, pos_2d, masked, translation, rotation, w,
                    d_3d, loss_reproj, loss_3d, transform, translation])
            o_masked_sqr = session.run(masked_sqr)
        # o_t, o_r = session.run([translation, rotation])
    # print("pos_3d: %s" % o_pos_3d)
    # print("pos_2d: %s" % o_pos_2d)
    # print("o_loss_reproj: %s, o_loss_3d: %s" % (o_loss_reproj, o_loss_3d))
    # print("t: %s" % o_t)
    # print("r: %s" % o_r)
    chosen = sorted((i for i in range(o_loss_reproj.shape[0])),
                    key=lambda i2: o_loss_reproj[i2] + o_loss_3d[i2])
    lg.info("Best candidate is %d with error %g + %g"
            % (chosen[0], o_loss_reproj[chosen[0]], o_loss_3d[chosen[0]]))
    # print("masked: %s" % o_masked)
    # opp = np.zeros_like(o_pos_3d)
    # for i in range(o_pos_3d.shape[0]):
    #     for j in range(o_pos_3d.shape[1]):
    #         for k in range(16):
    #             opp[i, j, :2, k] = o_pos_3d[i, j, :2, k] / o_pos_3d[i, j, 2:3, k]
    #             # opp[i, j, 0, k] *= intr[0, 0]
    #             # opp[i, j, 1, k] *= intr[1, 1]
    #             # opp[i, j, :2, k] *= intr[1, 1]
    #             a = o_pos_2d[i, j, :, k]
    #             b = opp[i, j, :2, k]
    #             if not np.allclose(a, b):
    #                 print("diff: %s, %s" % (a, b))

    o_pos_2d[:, :, 0, :] *= intr[0, 0]
    o_pos_2d[:, :, 1, :] *= intr[1, 1]
    o_pos_2d += intr[:2, 2:3]

    # for cand_id in range(o_pos_2d.shape[0]):
    if False:
        # return
        # print("w: %s" % o_w)
        # print("conf_2d: %s" % conf_2d)
        # lg.debug("query_2d[0, 0, ...]: %s" % query_2d[0, 0, ...])
        query_2d[:, 0, :] *= intr[0, 0]
        query_2d[:, 1, :] *= intr[1, 1]
        # lg.debug("query_2d[0, 0, ...]: %s" % query_2d[0, 0, ...])
        query_2d += intr[:2, 2:3]
        # lg.debug("query_2d[0, 0, ...]: %s" % query_2d[0, 0, ...])

        ims = {}
        for cand_id in chosen[:5]:
            lg.debug("starting %s" % cand_id)
            pos_ = o_pos_2d[cand_id, ...]
            for lin_id in range(pos_.shape[0]):
                frame_id = gap[0] + lin_id
                try:
                    im = ims[frame_id].copy()
                except KeyError:
                    p_im = pjoin(d_query, 'origjpg',
                                 "color_%05d.jpg" % frame_id)
                    ims[frame_id] = cv2.imread(p_im)
                    im = ims[frame_id].copy()
                # im = im_.copy()
                for jid in range(pos_.shape[-1]):

                    xy2 = int(round(query_2d[lin_id, 0, jid])), \
                          int(round(query_2d[lin_id, 1, jid]))
                    # print("printing %s" % repr(xy))
                    cv2.circle(im, center=xy2, radius=5, color=(10., 200., 10.),
                               thickness=-1)

                    if o_masked[cand_id, lin_id, 0, jid] > 0 \
                       or o_w[lin_id, 0, jid] > 0:
                        xy = int(round(pos_[lin_id, 0, jid])), \
                             int(round(pos_[lin_id, 1, jid]))
                        # print("printing %s" % repr(xy))
                        cv2.circle(im, center=xy, radius=3, color=(200., 10., 10.),
                                   thickness=-1)
                        cv2.putText(
                            im,
                            "d2d: %g" % o_masked_sqr[cand_id, lin_id, jid],
                            org=((xy2[0] - xy[0]) // 2 + xy[0],
                                 (xy2[1] - xy[1]) // 2 + xy[1]),
                            fontFace=1, fontScale=1, color=(0., 0., 0.))
                        cv2.line(im, xy, xy2, color=(0., 0., 0.))
                        d3d = o_d_3d[cand_id, lin_id, :, jid]
                        d3d_norm = np.linalg.norm(d3d)
                        if d3d_norm > 0.:
                            cv2.putText(im, "%g" % d3d_norm,
                                        org=((xy2[0] - xy[0]) // 2 + xy[0] + 10,
                                             (xy2[1] - xy[1]) // 2 + xy[1]),
                                        fontFace=1, fontScale=1,
                                        color=(0., 0., 255.))

                cv2.putText(im, text="%d::%02d" % (cand_id, lin_id),
                            org=(40, 80), fontFace=1, fontScale=2,
                            color=(255., 255., 255.)
                            )

                # pos_2d_ = np.matmul(intr, pos_[lin_id, :2, :] / pos_[lin_id, 2:3, :])
                # for p2d in pos_2d_
                cv2.imshow('im', im)
                cv2.waitKey()
            break

        while cv2.waitKey() != 27:
            pass

    out_scenelets = []
    for cand_id in chosen[:1]:
        lg.debug("score of %d is %g + %g = %g"
                 % (cand_id, o_loss_reproj[cand_id],
                    o_loss_3d[cand_id],
                    o_loss_reproj[cand_id] + o_loss_3d[cand_id]))
        scenelet = Scenelet()
        rate = query_full.skeleton.get_rate()
        prev_time = None
        for lin_id, frame_id in enumerate(range(gap[0], gap[1] + 1)):
            time_ = query_full.get_time(frame_id)
            if lin_id and rate is None:
                rate = time_ - prev_time
            if time_ == frame_id:
                time_ = prev_time + rate
            scenelet.skeleton.set_pose(frame_id=frame_id,
                                       pose=o_pos_3d[cand_id, lin_id, :, :],
                                       time=time_)
            prev_time = time_
        tr = np.concatenate((
            np.concatenate((
                o_transform[cand_id, ...],
                o_translation[cand_id, None, :].T), axis=1),
            [[0., 0., 0., 1.]]
        ), axis=0)
        tr_m = np.concatenate((
            np.concatenate((np.identity(3), -sclt_means[cand_id, None, :].T),
                           axis=1),
            [[0., 0., 0., 1.]]
        ), axis=0)
        tr = np.matmul(tr, tr_m)
        for oid, ob in scene.objects.items():
            if ob.label in ('wall', 'floor'):
                continue
            ob2 = copy.deepcopy(ob)
            ob2.apply_transform(tr)
            scenelet.add_object(obj_id=oid, scene_obj=ob2, clone=False)
        scenelet.name_scene = scene.name_scene
        out_scenelets.append((o_loss_reproj[cand_id], scenelet))
    return out_scenelets


def main(argv):
    np.set_printoptions(suppress=True, linewidth=200)
    pjoin = os.path.join

    parser = argparse.ArgumentParser("matcher")
    parser.add_argument("d_scenelets", type=argparse_check_exists,
                        help="Folder containing scenelets")
    parser.add_argument("video", type=argparse_check_exists,
                        help="Input path")
    parser.add_argument("--gap-size-limit", type=int,
                        help="Smallest gap size to still explain")
    args = parser.parse_args(argv)
    d_query = os.path.dirname(args.video)

    # 2d keypoint rescale
    p_im = pjoin(d_query, 'origjpg', 'color_00100.jpg')
    im_ = cv2.imread(p_im)
    shape_orig = im_.shape
    scale_2d = shape_orig[0] / float(INPUT_SIZE)

    query = Scenelet.load(args.video, no_obj=True)
    tr_ground = np.array(query.aux_info['ground'], dtype=np.float32)
    print("tr: %s" % tr_ground)

    name_query = os.path.basename(args.video).split('_')[1]
    query_2d = Scenelet.load(pjoin(d_query,
                                   "skel_%s_2d_00.json" % name_query)).skeleton
    p_intr = pjoin(d_query, 'intrinsics.json')
    intr = np.array(json.load(open(p_intr, 'r')), dtype=np.float32)
    lg.debug("intr: %s" % intr)

    gaps = find_gaps(query.skeleton, min_pad=1)

    p_scenelets_pickle = pjoin(args.d_scenelets, 'match_gap_scenelets.pickle')
    if os.path.exists(p_scenelets_pickle):
        scenelets = pickle_load(open(p_scenelets_pickle, 'rb'))
    else:
        scenelets = read_scenelets(args.d_scenelets, limit=0)
        pickle.dump(scenelets, open(p_scenelets_pickle, 'wb'))

    p_out_sclts = pjoin(d_query, 'fill')
    if os.path.exists(p_out_sclts):
        shutil.rmtree(p_out_sclts)
    os.makedirs(p_out_sclts)
    times = []
    for gap_id, gap in enumerate(gaps):
        lg.debug("gap is %s" % repr(gap))
        if gap[1] - gap[0] < args.gap_size_limit:
            continue
        with Timer("gap %d" % gap_id) as timer:
            chosen = []
            for sc_id, sclt in enumerate(scenelets):
                lg.info("scenelet %d / %d" % (sc_id, len(scenelets)))
                sclt = scenelets[sc_id]
                ground_obj = next(ob for ob in sclt.objects.values()
                              if ob.label == 'floor')
                ground_part = ground_obj.get_part_by_name('floor')
                lg.debug("floor: %s" % ground_part)
                ground_transform = ground_part.obb.as_transform()
                lg.debug("floor: %s" % ground_transform)
                # sys.exit(0)
                # chosen.extend(
                out_sclts = match(query, d_query, query_2d, sclt, intr, gap,
                                  tr_ground, scale=scale_2d)
                if not len(out_sclts):
                    continue
                # pick best from scene
                chosen.append([out_sclts[0][i] for i in range(len(out_sclts[0]))]
                              + [sc_id])
                # break

            # os.system("rm %s/skel_%s_fill_%03d_%03d__*.json"
            #           % (p_out_sclts, name_query, gap[0], gap[1]))
            chosen = sorted(chosen, key=lambda score_sclt: score_sclt[0])
            for sid, (score, out_sclt, sc_id) in enumerate(chosen):
                p_out = pjoin(p_out_sclts,
                              "skel_%s_fill_%03d_%03d__%02d.json"
                              % (name_query, gap[0], gap[1], sid))
                out_sclt.save(p_out)
                if sid > 5:
                    break
        times.append(timer.get_elapsed_ms())
    lg.info("mean time per gap: %s" % np.mean(times))


if '__main__' == __name__:
    main(sys.argv[1:])