import copy

from imapper.logic.scenelet import Scenelet
from imapper.visualization.plotting import plt

import cv2
import numpy as np
import scipy.stats
import sys
# print(sys.path)
# sys.path.append("/usr/local/lib/python3.6/site-packages/")
import tensorflow as tf
# sys.path.pop(-1)
# print(sys.path)
from matplotlib.lines import Line2D
from scipy.signal import medfilt
from tensorflow.contrib.opt import ScipyOptimizerInterface
from enum import IntEnum

from imapper.config.conf import Conf
from imapper.logic.joints import Joint
from imapper.util.stealth_logging import lg
from imapper.util.timer import Timer

# draw options
JOINT_DRAW_SIZE = 3
LIMB_DRAW_SIZE = 2
_LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13]).reshape((-1, 2))
_COLORS = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
           [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]
# draw options
JOINT_DRAW_SIZE = 3
LIMB_DRAW_SIZE = 2

# lg.debug(os.environ['LD_LIBRARY_PATH'])
# if 'gurobi' not in os.environ['LD_LIBRARY_PATH']:
#     raise RuntimeError("Need gurobi")


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculates the huber loss.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg * (err - .5 * mg)
    quad = .5 * err * err
    return tf.where(err < mg, quad, lin)


def plot_2d(pose_2d, image):
    """Plot our poses"""
    plt.imshow(image)
    # pose_2d = skel_ours_2d.get_pose(frame_id)
    for jid in xrange(pose_2d.shape[1]):
        print("pose_2d[%d]: %s" % (jid, pose_2d[0, jid]))
        plt.scatter(pose_2d[0, jid], pose_2d[1, jid], s=20)
        plt.text(pose_2d[0, jid], pose_2d[1, jid], '%s' % (str(jid)),
                 size=20, zorder=1, color='k')


def filter_wrong_poses(
  skel_ours_2d, skel_ours_3d,
  d_thresh=Conf.get().optimize_path.head_ank_dthresh,
  l_torso_thresh=Conf.get().optimize_path.torso_length_thresh,
  show=False):
    """Attempts to filter poses, where ankles are too close to the head.
    Remember, up is -y, so lower y coordinate means "higher" in space.

    TODO: fix transition between multiple actors
    """

    if show:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)

    y0 = []
    y1 = []
    l_torsos = []
    d = []
    x = []
    frames = np.array(skel_ours_2d.get_frames(), dtype='i8')
    frames.flags.writeable = False
    for frame_id in frames:
        pose = skel_ours_3d.get_pose(frame_id=frame_id)
        x.append(frame_id)
        top = np.min(pose[1, [Joint.HEAD, Joint.LSHO, Joint.RSHO]])
        bottom = min(pose[1, Joint.LANK], pose[1, Joint.RANK])
        y0.append(-top)
        y1.append(-bottom)
        l_torso = np.linalg.norm(pose[:, Joint.THRX] - pose[:, Joint.PELV])
        l_torsos.append(l_torso)
        # d.append(-(top - bottom))
    l_torsos = np.array(l_torsos)
    y0_f = medfilt(y0, 3)
    y1_f = medfilt(y1, 3)
    dt_b = np.logical_and(
        np.concatenate(([True], np.less(frames[1:] - frames[:-1], 2))),
        np.concatenate((np.less(frames[:-1] - frames[1:], 2), [True]))
    )
    y0 = np.where(dt_b, y0_f, y0)
    y1 = np.where(dt_b, y1_f, y1)
    d = (y0 - y1)
    x = np.array(x, dtype='i8')
    frames_to_remove = x[d < d_thresh]
    min_y = min(np.min(y0), np.min(y1))
    y0 -= min_y
    y1 -= min_y

    if show:
        ax.plot(x, y0, 'g', label='max(head, *sho)')
        ax.plot(x, y1, 'b', label='min(*ank)')
        ax.plot(x, d, 'm', label='diff')
        for frame_id in frames_to_remove:
            ax.scatter(
                frame_id,
                skel_ours_3d.get_joint_3d(Joint.LANK, frame_id=frame_id)[1],
                c='r', marker='o'
            )
        ax.legend(bbox_to_anchor=(1.01, 1))
        ax.set_title("3D Local space joint positions over time")
        ax.set_xlabel("Discrete time")
        ax.set_ylabel("Height of joints")

        ax = fig.add_subplot(122)
        ax.plot(x, l_torsos)
        plt.subplots_adjust(right=0.7, wspace=0.5)
    frames_to_remove = list(set(
        np.append(frames_to_remove, x[l_torsos < l_torso_thresh]).tolist()))
    if show:
        for frame_id in frames_to_remove:
            ax.scatter(
                frame_id,
                skel_ours_3d.get_joint_3d(Joint.LANK, frame_id=frame_id)[1],
                c='r', marker='o'
            )
        plt.show()
        # plt.pause(100)
        plt.close()
        # sys.exit()

    # for frame_id in frames_to_remove:
    #     skel_ours_2d.remove_pose(frame_id)

    return frames_to_remove #, skel_ours_2d


def filter_outliers(skel_ours_2d, winsorize_limit=0.05,
                    show=False):
    lg.debug('[filter_outliers] Filtering based on 2D displacement...')
    seq = skel_ours_2d.poses[:, :2, :]
    displ = seq[1:, :, :] - seq[:-1, :, :]
    frames = np.array(skel_ours_2d.get_frames_mod())
    assert displ.shape[0] == seq.shape[0] - 1 \
        and displ.shape[1] == seq.shape[1] \
        and displ.shape[2] == seq.shape[2], \
        "No: %s" % displ.shape
    displ = np.linalg.norm(displ, axis=1)
    assert displ.shape == (seq.shape[0] - 1, seq.shape[2]), \
        "No: %s" % repr(displ.shape)

    if show:
        plt.figure()
    # Ensure we only filter consequtive frames
    dtime = frames[1:] - frames[:-1]
    # lg.debug("delta time: %s" % dtime)

    out = scipy.stats.mstats.winsorize(displ, limits=winsorize_limit)
    # diff = np.linalg.norm(out - displ, axis=1)
    diff = out - displ
    fraction_corrected = np.sum((diff < -1.).astype('i4'), axis=1) \
                         / float(diff.shape[1])
    # print("diff: %s" % diff)

    # threshold changed from 5. to 0.6 on 19/1/2018
    # threshold changed from 0.6 to 0.5 on 20/1/2018
    lin_ids_to_remove = np.argwhere(fraction_corrected > 0.6)
    frame_ids_to_remove = \
        [skel_ours_2d.get_frame_id_for_lin_id(lin_id)
         for lin_id in np.squeeze(lin_ids_to_remove, axis=1).tolist()
         if dtime[lin_id] == 1
         and (lin_id+1 >= dtime.size or dtime[lin_id+1] == 1)]
    cpy = copy.deepcopy(skel_ours_2d)
    for frame_id in frame_ids_to_remove:
        lg.debug("Removing frame_id %d because it jumped in 2D."
                 % frame_id)
        skel_ours_2d.remove_pose(frame_id)
        for frame_id_ in skel_ours_2d._frame_ids.keys():
            if frame_id_ != frame_id:
                assert np.allclose(skel_ours_2d.get_pose(frame_id_),
                                   cpy.get_pose(frame_id_)), \
                    "No (frame_id: %d, lin_id: %d, old lin_id: %d)\nnew: %s\nold:\n%s" \
                    % (frame_id_, skel_ours_2d.get_lin_id_for_frame_id(frame_id_),
                       cpy.get_lin_id_for_frame_id(frame_id_),
                       skel_ours_2d.get_pose(frame_id_),
                       cpy.get_pose(frame_id_)
                       )

    # mask = np.where(diff < 10., True, False)
    # if mask[-1]:
    #     mask = np.append(mask, mask[-1])
    # else:
    #     mask = np.insert(mask, 0, mask[0])
    # print(mask)
    # skel_ours_2d._poses = skel_ours_2d._poses[mask, :, :]
    # assert list(skel_ours_2d._frame_ids.values()) == sorted(list(skel_ours_2d._frame_ids.values())), \
    #     "Not sorted: %s" % list(skel_ours_2d._frame_ids.values())
    # skel_ours_2d._frame_ids = dict((k, v) for i, (k, v) in enumerate(skel_ours_2d._frame_ids.items()) if mask[i])
    # assert list(skel_ours_2d._frame_ids.values()) == sorted(list(skel_ours_2d._frame_ids.values())), \
    #     "Not sorted: %s" % list(skel_ours_2d._frame_ids.values())
    if show:
        plt.plot(diff, 'k')
        plt.plot(displ[:, 6], 'g')
        plt.plot(out[:, 6], 'b')
        plt.show()
    return skel_ours_2d, frame_ids_to_remove


class SmoothMode(IntEnum):
    """Path smoothing in 3D type flag."""

    ACCEL = 0
    """Smooth acceleration, e.g. change of velocity (second order)."""
    VELOCITY = VEL = 1
    """Smooth velocity, e.g. path length (first order)."""

    @classmethod
    def from_string(cls, string):
        return getattr(cls, string.upper(), None)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def debug_forwards(ip, op, rot, forwards, iangles):
    import os
    import shutil
    from mpl_toolkits import mplot3d
    from stealth.logic.skeleton import Skeleton
    from alignment import get_angle

    dest = 'tmp'
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    assert ip.shape[0] == op.shape[0], "ip.shape: {}, op.shape: {}".format(ip.shape, op.shape)
    assert ip.shape[0] == len(forwards), 'ip.shape: {}, nfws: {}'.format(ip.shape, len(forwards))
    c0, c1, c2 = 0, 2, 1
    colors = [(1., 0., 0.), (0., 1., 0)]
    unit_x = np.array((1., 0., 0.))
    for lin_id in range(ip.shape[0]):
        ip0 = ip[lin_id, :, :]
        op0 = op[lin_id, :, :]
        f = plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(azim=0, elev=80)

        ifw = Skeleton.get_forward_from_pose(ip0)
        ofw = Skeleton.get_forward_from_pose(op0)
        ang = angle_between(ifw, ofw)
        iang = get_angle(ifw, unit_x)
        oang = -np.arctan2(ofw[2], ofw[0]) 
        fws = [ifw, ofw]
        print(ifw, ofw)
        for i, sk in enumerate([ip0, op0]):
            ax.scatter(
                sk[c0, :], 
                sk[c1, :], 
                -sk[c2, :], 
                c=(0.5, 0.5, 0.5), 
                linewidth=0.5)
            
            for limb in [[Joint.HEAD, Joint.THRX], [Joint.LHIP, Joint.RHIP],
                         [Joint.LKNE, Joint.LANK], [Joint.RKNE, Joint.RANK],
                         [Joint.LHIP, Joint.LKNE], [Joint.RHIP, Joint.RKNE], 
                         [Joint.LSHO, Joint.THRX], [Joint.RSHO, Joint.THRX], 
                         [Joint.THRX, Joint.PELV], [Joint.LSHO, Joint.LELB],
                         [Joint.RSHO, Joint.RELB], [Joint.RELB, Joint.RWRI],
                         [Joint.LELB, Joint.LWRI]
                         ]:
                ax.plot( 
                    [sk[c0, limb[0]], sk[c0, limb[1]]],
                    [sk[c1, limb[0]], sk[c1, limb[1]]],
                    [-sk[c2, limb[0]], -sk[c2, limb[1]]], 
                    c=colors[i]
                )
                ax.plot(
                    [ sk[c0, Joint.HEAD],  sk[c0, Joint.HEAD] + fws[i][c0]],
                    [ sk[c1, Joint.HEAD],  sk[c1, Joint.HEAD] + fws[i][c1]],
                    [-sk[c2, Joint.HEAD], -sk[c2, Joint.HEAD] - fws[i][c2]],
                    c=(1., 1., 0.)
                )
        ax.set_title('rot: {:.2f}, dang: {:.2f}, rot0: {:.2f}, rot1: {:.2f}'.format(
            np.rad2deg(rot[lin_id][1]), np.rad2deg(ang), np.rad2deg(iang), np.rad2deg(oang)))
        plt.savefig(os.path.join(dest, 'rot_{:5d}.jpg'.format(lin_id)))
        plt.close()
    sys.exit(1)


def optimize_path(skel_ours, skel_ours_2d, images, intrinsics,
                  path_skel, ground_rot, shape_orig=None,
                  use_huber=False, weight_smooth=0.01,
                  show=False, frames_ignore=None,
                  resample=True, depth_init=10.,
                  p_constraints=None, smooth_mode=SmoothMode.ACCEL):
    """Optimize 3D path so that it matches the 2D corresponding observations.

    Args:
        skel_ours (Skeleton):
            3D skeleton from LFD.
        skel_ours_2d (Skeleton):
            2D feature points from LFD.
        images (dict):
            Color images for debug, keyed by frame_ids.
        camera_name (str):
            Initialize intrinsics matrix based on name of camera.
        path_skel (str):
            Path of input file from LFD on disk, used to create paths for
            intermediate result.
        shape_orig (tuple):
            Height and width of original images before LFD scaled them.
        use_huber (bool):
            Deprecated.
        weight_smooth (float):
            Smoothness term weight.
        winsorize_limit (float):
            Outlier detection parameter.
        show (bool):
            Show debug visualizations.
        frames_ignore (set):
            Deprecated.
        resample (bool):
            Fill in missing poses by interpolating using Blender's IK.
        depth_init (float):
            Initial depth for LFD poses.
        p_constraints (str):
            Path to 3D constraints scenelet file.
        smooth_mode (SmoothMode):
            Smooth velocity or acceleration.
    """

    # scale 2D detections to canonical camera coordinates
    np_poses_2d = \
        skel_ours_2d.poses[:, :2, :] \
        - np.expand_dims(intrinsics[:2, 2], axis=1)
    np_poses_2d[:, 0, :] /= intrinsics[0, 0]
    np_poses_2d[:, 1, :] /= intrinsics[1, 1]

    n_frames = skel_ours.poses.shape[0]
    np_translation = np.zeros(shape=(n_frames, 3), dtype=np.float32)
    np_translation[:, 1] = -1.
    np_translation[:, 2] = \
        np.random.uniform(-depth_init * 0.25, depth_init * 0.25,
                          np_translation.shape[0]) \
        + depth_init
    np_rotation = np.zeros(shape=(n_frames, 3), dtype=np.float32)

    frame_ids = np.array(skel_ours.get_frames(), dtype=np.float32)
    np_visibility = skel_ours_2d.get_confidence_matrix(
      frame_ids=frame_ids, dtype='f4')

    if p_constraints is not None:
        sclt_cnstr = Scenelet.load(p_constraints)
        np_cnstr_mask = np.zeros(
          shape=(len(frame_ids), Joint.get_num_joints()), dtype=np.float32)
        np_cnstr = np.zeros(shape=(len(frame_ids), 3, Joint.get_num_joints()),
                            dtype=np.float32)
        for frame_id, confs in sclt_cnstr.confidence.items():
            lin_id = None
            for j, conf in confs.items():
                if conf > 0.5:
                    if lin_id is None:
                        lin_id = next(
                            lin_id_
                            for lin_id_, frame_id_ in enumerate(frame_ids)
                            if frame_id_ == frame_id)
                    np_cnstr_mask[lin_id, j] = conf
                    np_cnstr[lin_id, :, j] = \
                        sclt_cnstr.skeleton.get_joint_3d(
                          joint_id=j, frame_id=frame_id)
    else:
        np_cnstr_mask = None
        np_cnstr = None

    spans = skel_ours.get_actor_empty_frames()
    dt = frame_ids[1:].astype(np.float32) \
         - frame_ids[:-1].astype(np.float32)
    dt_pos_inv = np.reciprocal(dt, dtype=np.float32)
    dt_vel_inv = np.divide(np.float32(2.), dt[1:] + dt[:-1])
    # ensure smoothness weight multipliers are not affected by
    # actor-transitions
    if skel_ours.n_actors > 1 and len(spans):
        for lin_id in range(len(dt)):
            frame_id0 = frame_ids[lin_id]
            frame_id1 = frame_ids[lin_id + 1]
            span = next((span_ for span_ in spans
                         if span_[0] == frame_id0),
                        None)
            if span is not None:
                assert frame_id1 == span[1], "No"
                dt[lin_id] = 0.
                dt_pos_inv[lin_id] = 0.
                dt_vel_inv[lin_id] = 0.
                dt_vel_inv[lin_id - 1] = 1. / dt[lin_id-1]

    forwards = np.array([skel_ours.get_forward(frame_id, estimate_ok=True, k=0) 
                for frame_id in skel_ours.get_frames()])
    # from alignment import get_angle
    # xs = np.hstack((
            # np.ones(shape=(len(forwards), 1)), 
            # np.zeros(shape=(len(forwards), 2))
        # ))
    # print(xs.shape)
    print(forwards.shape)
    unit_x = np.array((1., 0., 0.))
    np_angles = [-np.arctan2(forward[2], forward[0])
                 for forward in forwards]
    print(forwards, np_angles)
    # ank_diff = \
    #     np.exp(
    #        -2. * np.max(
    #           [
    #               np.linalg.norm(
    #                  (skel_ours.poses[1:, :, joint]
    #                   - skel_ours.poses[:-1, :, joint]).T
    #                  * dt_pos_inv, axis=0
    #               ).astype(np.float32)
    #               for joint in {Joint.LANK, Joint.RANK}
    #           ],
    #           axis=0
    #        )
    #     )
    # assert ank_diff.shape == (skel_ours.poses.shape[0]-1,), \
    #     "Wrong shape: %s" % repr(ank_diff.shape)

    # cam_angle = [np.deg2rad(-8.)]
    assert np.isclose(ground_rot[1], 0.) and np.isclose(ground_rot[2], 0.), \
        "Assumed only x rotation"
    # assert ground_rot[0] <= 0, "Negative means looking down, why looknig up?"
    cam_angle = [np.deg2rad(ground_rot[0])]
    # assert False, "Fixed angle!"
    device_name = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
    devices = {device_name}
    for device in devices:
        with Timer(device, verbose=True):
            graph = tf.Graph()
            with graph.as_default(), tf.device(device):
                tf_visibility = tf.Variable(np.tile(np_visibility, (1, 2, 1)),
                                            name='visibility', trainable=False,
                                            dtype=tf.float32)
                tf_dt_pos_inv = \
                    tf.Variable(np.tile(dt_pos_inv, (1, 3)).reshape(-1, 3),
                                name='dt_pos_inv', trainable=False,
                                dtype=tf.float32)
                tf_dt_vel_inv = \
                    tf.constant(np.tile(dt_vel_inv, (1, 3)).reshape(-1, 3),
                                name='dt_vel_inv', dtype=tf.float32)

                # input data
                pos_3d_in = tf.Variable(skel_ours.poses.astype(np.float32),
                                        trainable=False, name='pos_3d_in',
                                        dtype=tf.float32)
                pos_2d_in = tf.Variable(np_poses_2d.astype(np.float32),
                                        trainable=False, name='pos_2d_in',
                                        dtype=tf.float32)

                params_camera = tf.Variable(initial_value=cam_angle,
                                            dtype=tf.float32,
                                            trainable=True)

                cam_sn = tf.sin(params_camera)
                cam_cs = tf.cos(params_camera)
                transform_camera = tf.reshape(
                  tf.stack(
                    [1., 0., 0., 0.,
                     0., cam_cs[0], cam_sn[0], 0.,
                     0., -cam_sn[0], cam_cs[0], 0.,
                     0., 0., 0., 1.],
                    axis=0),
                  shape=(4, 4))

                # 3D translation
                translation = tf.Variable(np_translation, name='translation')
                # 3D rotation (Euler XYZ)
                rotation = tf.Variable(np_rotation, name='rotation')
                fw_angles = tf.Variable(np_angles, name='angles')

                # rotation around y
                my_zeros = tf.zeros((n_frames, 1))
                my_ones = tf.ones((n_frames, 1))
                c = tf.cos(tf.slice(rotation, [0, 1], [n_frames, 1]))
                s = tf.sin(tf.slice(rotation, [0, 1], [n_frames, 1]))
                t0 = tf.concat([c, my_zeros, -s, my_zeros], axis=1)
                t1 = tf.concat([my_zeros, my_ones, my_zeros, my_zeros], axis=1)
                t2 = tf.concat([s, my_zeros, c, my_zeros], axis=1)
                t3 = tf.concat([my_zeros, my_zeros, my_zeros, my_ones], axis=1)
                transform = tf.stack([t0, t1, t2, t3], axis=2,
                                     name="transform")

                transform = tf.einsum('ij,ajk->aik',
                                      transform_camera,
                                      transform)[:, :3, :3]

                # transform to 3d
                pos_3d = tf.matmul(transform, pos_3d_in) \
                    + tf.tile(tf.expand_dims(translation, 2),
                              [1, 1, int(pos_3d_in.shape[2])])

                # constraints
                loss_cnstr = None
                if np_cnstr is not None:
                    constraints = tf.Variable(
                      np_cnstr, trainable=False, name='constraints',
                      dtype=tf.float32)
                    constraints_mask = tf.Variable(
                      np_cnstr_mask, trainable=False, name='constraints_mask',
                      dtype=tf.float32)
                    cnstr_diff = tf.reduce_sum(
                      tf.squared_difference(pos_3d, constraints), axis=1,
                      name='constraints_difference')
                    cnstr_diff_masked = tf.multiply(
                      constraints_mask, cnstr_diff,
                      name='constraints_difference_masked')
                    loss_cnstr = tf.reduce_sum(cnstr_diff_masked,
                                               name='constraints_loss')

                # perspective divide
                pos_2d = tf.divide(
                    tf.slice(pos_3d, [0, 0, 0], [n_frames, 2, -1]),
                    tf.slice(pos_3d, [0, 2, 0], [n_frames, 1, -1]))

                if use_huber:
                    diff = huber_loss(pos_2d_in, pos_2d, 1.)
                    masked = diff * tf_visibility
                    loss_reproj = tf.nn.l2_loss(masked)
                    lg.info("Doing huber on reprojection, NOT translation")
                else:
                    # re-projection loss
                    diff = pos_2d - pos_2d_in
                    # mask loss by 2d key-point visibility
                    masked = diff * tf_visibility
                    loss_reproj = tf.nn.l2_loss(masked)
                    lg.info("NOT doing huber")

                sys.stderr.write(
                    "TODO: Move huber to translation, not reconstruction\n")

                # translation smoothness
                dx = tf.multiply(
                  x=0.5,
                  y=tf.add(
                    pos_3d[1:, :, Joint.LHIP] - pos_3d[:-1, :, Joint.LHIP],
                    pos_3d[1:, :, Joint.RHIP] - pos_3d[:-1, :, Joint.RHIP],
                    ),
                  name="average_hip_displacement_3d"
                )
                tf_velocity = tf.multiply(dx, tf_dt_pos_inv)

                tf_acceleration_z = tf.multiply(x=dx[1:, 2:3] - dx[:-1, 2:3],
                                                y=tf_dt_vel_inv[:, 2:3],
                                                name="acceleration_z")

                if smooth_mode == SmoothMode.VELOCITY:
                    # if GT, use full smoothness to fix 2-frame flicker
                    if np_cnstr is not None:
                        print('Smoothing all velocity!')
                        loss_transl_smooth = \
                            weight_smooth * tf.nn.l2_loss(tf_velocity)
                    else: # Normal mode, don't oversmooth screen-space
                        loss_transl_smooth = \
                            weight_smooth * tf.nn.l2_loss(tf_velocity[:, 2:3])
                elif smooth_mode == SmoothMode.ACCEL:
                    loss_transl_smooth = \
                        weight_smooth * tf.nn.l2_loss(tf_acceleration_z)
                else:
                    raise RuntimeError('Unknown smooth mode: {}'
                                       .format(smooth_mode))

                if show:
                    sqr_accel_z = weight_smooth * tf.square(tf_acceleration_z)

                if weight_smooth > 0.:
                    lg.info("Smoothing in time!")
                    loss = loss_reproj + loss_transl_smooth
                else:
                    lg.warning("Not smoothing!")
                    loss = loss_reproj

                if loss_cnstr is not None:
                    loss += 1000 * loss_cnstr

                # hip0 = tf.nn.l2_normalize(pos_3d[:-1, :, Joint.RHIP] - pos_3d[:-1, :, Joint.LHIP])
                # hip1 = tf.nn.l2_normalize(pos_3d[1:, :, Joint.RHIP] - pos_3d[1:, :, Joint.RHIP])
                # dots = tf.reduce_sum(tf.multiply(hip0, hip1), axis=1)
                # print(dots)
                # loss_dot = tf.nn.l2_loss(1. - dots)
                # loss_ang = fw_angles + rotation[:, 1]
                # print(loss_ang)
                # loss_ang = tf.square(loss_ang[1:] - loss_ang[:-1])
                # print(loss_ang)
                # two_pi_sqr = tf.constant((2. * 3.14159)**2., dtype=tf.float32)
                # print(two_pi_sqr)
                # loss_ang = tf.reduce_mean(tf.where(loss_ang > two_pi_sqr, loss_ang - two_pi_sqr, loss_ang))
                # print(loss_ang)
                # loss += loss_ang

                #
                # optimize
                #
                optimizer = ScipyOptimizerInterface(
                   loss, var_list=[translation, rotation],
                   options={'gtol': 1e-12},
                   var_to_bounds={rotation: (-np.pi/2., np.pi/2.)}
                )

            with tf.Session(graph=graph) as session:
                session.run(tf.global_variables_initializer())

                optimizer.minimize(session)
                np_pos_3d_out, np_pos_2d_out, np_transl_out, np_masked, \
                np_acceleration, np_loss_transl_smooth, np_dt_vel = \
                    session.run([pos_3d, pos_2d, translation, masked,
                                 tf_acceleration_z, loss_transl_smooth,
                                 tf_dt_vel_inv])
                if show:
                    o_sqr_accel_z = session.run(sqr_accel_z)
                o_vel = session.run(tf_velocity)
                o_dx = session.run(dx)
                o_rot = session.run(rotation)
                # o_dx, o_dx2 = session.run([accel_bak, acceleration2])
                # assert np.allclose(o_dx, o_dx2), "no"
                o_cam = session.run(fetches=[params_camera])
                print("camera angle: %s" % np.rad2deg(o_cam[0]))
                # o_losses = session.run([loss_reproj, loss_transl_smooth, loss_dot, loss_ang])
                o_losses = session.run([loss_reproj, loss_transl_smooth])
                print('losses: {}'.format(o_losses))
                # o_dots = session.run(dots)
                # with open('tmp/dots.txt', 'w') as fout:
                #     fout.write('\n'.join((str(e) for e in o_dots.tolist())))

    fixed_frames = []
    # for lin_frame_id in range(np_transl_out.shape[0]):
    #     if np_transl_out[lin_frame_id, 2] < 0.:
    #         print("Correcting frame_id %d: %s"
    #               % (skel_ours.get_lin_id_for_frame_id(lin_frame_id),
    #                  np_transl_out[lin_frame_id, :]))
    #         if lin_frame_id > 0:
    #             np_transl_out[lin_frame_id, :] = np_transl_out[lin_frame_id-1, :]
    #         else:
    #             np_transl_out[lin_frame_id, :] = np_transl_out[lin_frame_id+1, :]
    #         fixed_frames.append(lin_frame_id)
    
    # debug_forwards(skel_ours.poses, np_pos_3d_out, o_rot, forwards, np_angles)

    # z_jumps = np_pos_3d_out[1:, 2, Joint.PELV] - np_pos_3d_out[:-1, 2, Joint.PELV]
    # out = scipy.stats.mstats.winsorize(z_jumps, limits=1.)
    # plt.figure()
    # plt.plot(pos_3d[:, 2, Joint.PELV])
    # plt.show()
    # sys.exit(0)
    # diff = np.linalg.norm(out - displ, axis=1)
    if len(fixed_frames):
        print("Re-optimizing...")
        with tf.Session(graph=graph) as session:
            np_pos_3d_out, np_pos_2d_out, np_transl_out = \
                session.run(fetches=[pos_3d, pos_2d, translation],
                            feed_dict={transform: np_transl_out})

    if show:
        lim_fr = [105, 115, 135]
        fig = plt.figure()
        accel_thr = 0.  # np.percentile(o_sqr_accel_z, 25)

        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        # print("np_masked:%s" % np_masked)
        # plt.plot(np_masked[:, )
        ax.plot(np.linalg.norm(
          np_acceleration[lim_fr[0]:lim_fr[1]],
          axis=1), '--o', label='accel')
        ax.add_artist(Line2D([0, len(o_sqr_accel_z)],
                             [accel_thr, accel_thr]))
        # plt.plot(np_dt_vel[:, 0], label='dt velocity')
        # plt.plot(np.linalg.norm(np_f_accel, axis=1), '--x', label='f_accel')
        # plt.plot(ank_diff, label='ank_diff')
        ax.plot(o_sqr_accel_z[lim_fr[0]:lim_fr[1]+1],
                '--x', label='loss accel_z')
        ax.legend()

        ax2 = plt.subplot2grid((2, 2), (1, 0), aspect='equal')
        ax2.plot(np_pos_3d_out[lim_fr[0]:lim_fr[1]+1, 0, Joint.PELV],
                 np_pos_3d_out[lim_fr[0]:lim_fr[1]+1, 2, Joint.PELV],
                 '--x')
        for i, vel in enumerate(o_vel):
            if not (lim_fr[0] <= i <= lim_fr[1]):
                continue

            p0 = np_pos_3d_out[i+1, [0, 2], Joint.PELV]
            p1 = np_pos_3d_out[i, [0, 2], Joint.PELV]
            ax2.annotate("%f = ((%g - %g) + (%g - %g)) * %g = %g"
                         % (vel[2],
                            np_pos_3d_out[i+1, 2, Joint.LHIP],
                            np_pos_3d_out[i, 2, Joint.LHIP],
                            np_pos_3d_out[i+1, 2, Joint.RHIP],
                            np_pos_3d_out[i, 2, Joint.RHIP],
                            np_dt_vel[i, 2],
                            o_dx[i, 2]),
                         xy=((p0[0] + p1[0]) / 2.,
                             (p0[1] + p1[1]) / 2.)
                         )
        ax2.set_title('velocities')

        ax1 = plt.subplot2grid((2, 2), (1, 1), aspect='equal')
        ax1.plot(np_pos_3d_out[lim_fr[0]:lim_fr[1]+1, 0, Joint.PELV],
                 np_pos_3d_out[lim_fr[0]:lim_fr[1]+1, 2, Joint.PELV],
                 '--x')
        for i, lacc in enumerate(o_sqr_accel_z):
            if not (lim_fr[0] <= i <= lim_fr[1]):
                continue
            if lacc > accel_thr:
                p0 = np_pos_3d_out[i+1, [0, 2], Joint.PELV]
                ax1.annotate("%.3f" % np_acceleration[i],
                             xy=(p0[0], p0[1]))
                ax.annotate("%.3f" % np.log10(lacc),
                            xy=(i - lim_fr[0],
                                abs(np_acceleration[i]))
                            )
        ax1.set_title('accelerations')

        plt.show()

    np.set_printoptions(linewidth=200)
    np_pos_2d_out[:, 0, :] *= intrinsics[0, 0]
    np_pos_2d_out[:, 1, :] *= intrinsics[1, 1]
    np_pos_2d_out[:, 0, :] += intrinsics[0, 2]
    np_pos_2d_out[:, 1, :] += intrinsics[1, 2]

    np_poses_2d[:, 0, :] *= intrinsics[0, 0]
    np_poses_2d[:, 1, :] *= intrinsics[1, 1]
    np_poses_2d[:, 0, :] += intrinsics[0, 2]
    np_poses_2d[:, 1, :] += intrinsics[1, 2]

    out_images = {}
    if shape_orig is not None:
        frames_2d = skel_ours_2d.get_frames()
        for frame_id2 in frames_2d:
            try:
                lin_frame_id = skel_ours_2d.get_lin_id_for_frame_id(frame_id2)
            except KeyError:
                lin_frame_id = None
            frame_id = skel_ours_2d.mod_frame_id(frame_id=frame_id2)

            im = None
            if frame_id in out_images:
                im = out_images[frame_id]
            elif len(images):
                if frame_id not in images:
                    lg.warning("Not enough images, the video was probably cut "
                               "after LiftingFromTheDeep was run.")
                    continue
                im = copy.deepcopy(images[frame_id])
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            else:
                im = np.zeros((shape_orig[0].astype(int),
                               shape_orig[1].astype(int), 3), dtype='i1')
            if lin_frame_id is not None:
                for jid in range(np_pos_2d_out.shape[2]):
                    if skel_ours_2d.is_visible(frame_id2, jid):
                        p2d = tuple(
                            np_pos_2d_out[lin_frame_id, :, jid]
                            .astype(int).tolist())
                        p2d_det = tuple(
                            np_poses_2d[lin_frame_id, :, jid]
                            .astype(int).tolist())
                        cv2.line(
                            im, p2d, p2d_det, color=(100, 100, 100), thickness=3)
                        cv2.circle(
                            im, p2d, radius=3, color=(0, 0, 200), thickness=-1)
                        cv2.circle(
                            im, p2d_det, radius=3, color=(0, 200, 0), thickness=-1)
            out_images[frame_id] = im
            # cv2.imshow("Out", im)
            # cv2.waitKey(50)

        if False:
            # visualize
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for frame_id in range(0, np_pos_3d_out.shape[0], 1):
                j = Joint.PELV
                ax.scatter(np_pos_3d_out[frame_id, 0, j],
                           np_pos_3d_out[frame_id, 2, j],
                           -np_pos_3d_out[frame_id, 1, j],
                           marker='o')
            # smallest = np_pos_3d_out.min()
            # largest = np_pos_3d_out.max()
            ax.set_xlim3d(-5., 5.)
            ax.set_xlabel('x')
            ax.set_ylim3d(-5., 5.)
            ax.set_ylabel('y')
            ax.set_zlim3d(-5., 5.)
            ax.set_zlabel('z')

    if False:
        # visualize
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for frame_id in range(0, np_pos_3d_out.shape[0], 1):
            for j in range(np_pos_3d_out.shape[2]):
                ax.scatter(np_pos_3d_out[frame_id, 0, j],
                           np_pos_3d_out[frame_id, 2, j],
                           -np_pos_3d_out[frame_id, 1, j],
                           marker='o')
        # smallest = np_pos_3d_out.min()
        # largest = np_pos_3d_out.max()
        ax.set_xlim3d(-5., 5.)
        ax.set_xlabel('x')
        ax.set_ylim3d(-5., 5.)
        ax.set_ylabel('y')
        ax.set_zlim3d(-5., 5.)
        ax.set_zlabel('z')
    plt.show()

    assert all(a == b
               for a, b in zip(skel_ours.poses.shape, np_pos_3d_out.shape)), \
        "no"
    skel_ours.poses = np_pos_3d_out
    return skel_ours, out_images, intrinsics
