import tensorflow as tf
import numpy as np

from imapper.logic.joints import Joint
from imapper.pose.point_poly_distance_estimator import \
    PointPolyDistanceEstimator
from imapper.pose.unk_manager import _FTYPE_NP
from imapper.util.stealth_logging import lg
from imapper.logic.categories import CATEGORIES

import sys
import pdb


def create_intersection_jo_losses(pos_3d, obj_2d_polys,
                                  joints_active,
                                  pos_3d_transform_indices,
                                  obj_2d_poly_transform_indices,
                                  independent, um, d_padding,
                                  cat_ids_polys):
    """Joint-object intersection loss creator.

    Currently only applied to the average of the hip joints.
    If obj_2d_poly_transform_indices are None, no masking is applied,
    e.g. all points intersect all polygons.

    Args:
        pos_3d (tf.Variable): (N, 14, 3)
            Transformed poses in world space 3D, permuted!
        obj_2d_polys (tf.Variable): (M, 4, 3)
            Transformed top-view polygons in world space 3D.
        joints_active (list):
            A list of joint names for the last dimension of pos_3d.
        pos_3d_transform_indices (tf.Variable): (N, 1)
            Transformation ids of the poses, indicating which are moving
            together. Can be None, in which case there will be no check for
            scenelet-self intersection (poses in scenelets should not be
            penalized for intersecting the objects in the scenelets,
            we assume them to be correct).
        obj_2d_poly_transform_indices (tf.Variable): (M, 4)
            Transformation ids of the polygons, indicating which are moving
            together. Can be None, see above.
        independent (bool):
            Quads from dynamic scenelets don't intersect poses from dynamic
            scenelets.
        d_padding (float):
            Extra padding around objects in meters.
    Returns:
        loss_jo (tf.Variable): (N,)
            A loss per input pose, which is the sum of squared penetration
            distances.
        loss_jo_per_transform (tf.Variable): (n_transforms, )
            A loss per transformation id.
    """
    dtype_tf = pos_3d.dtype
    dtype_np = dtype_tf.as_numpy_dtype

    assert pos_3d.get_shape().as_list()[2] == 3, "Assumed xyz in last dim"
    assert len(joints_active) == pos_3d.get_shape().as_list()[1], \
        "joints_active should describe the names of the joints in pos_3d:\n" \
        "%s vs %s" % (joints_active, pos_3d.get_shape().as_list())

    if Joint.LHIP not in joints_active or Joint.RHIP not in joints_active:
        raise RuntimeError("Need hips to compute intersection term.")

    # pelvis is hip average (no pelvis in features, so no pelvis in pos_3d)
    hips = tf.divide(tf.add(pos_3d[:, joints_active[Joint.LHIP], :],
                            pos_3d[:, joints_active[Joint.RHIP], :]),
                     dtype_np(2.))

    # It's enough, if all lower limbs are outside, sits have pelvis inside
    joints_tup = (
        hips,
        pos_3d[:, joints_active[Joint.LKNE], :],
        pos_3d[:, joints_active[Joint.RKNE]],
        # added 18/9/2018
        pos_3d[:, joints_active[Joint.LANK]],
        pos_3d[:, joints_active[Joint.RANK]]
    )
    joints = tf.concat(joints_tup, axis=0,
                       name="joint_object_intersection_joints")
    # joint-object distances
    d3 = PointPolyDistanceEstimator.point_poly_distance(
      p=joints,
      poly=obj_2d_polys,
      name='joint_object_distances_all')
    _n = hips.get_shape().as_list()[0]
    d3_2d = tf.reshape(d3, 
                       (len(joints_tup), _n, d3.get_shape().as_list()[1]),
                       name='joint_object_distances_per_pose')

    # max over distances, where negative distance means inside
    #  shape: (vertices, joints), e.g. (223, 3)
    # d = tf.reduce_max(d3_2d, axis=0, name='joint_object_distances_raw')

    d_all_outside = tf.reduce_min(d3_2d[1:, ...], axis=0, 
                                  name='knees_ankles_to_object_distances_raw')
    # pdb.set_trace()
    d_for_tables = d3_2d[0, ...]
    d_for_other = tf.minimum(d3_2d[0, ...], d_all_outside,
                             name='joint_object_distances_raw')
    CAT_TABLE = CATEGORIES['table']
    is_table_tiled = tf.tile(tf.equal(cat_ids_polys, CAT_TABLE),
                             multiples=(_n, 1), name='is_table_tiled')
    d = tf.where(condition=is_table_tiled, x=d_for_tables, y=d_for_other)

    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     print("_n: %s" % _n)
    #     pdb.set_trace()
    #     sys.exit(0)
    #     od3_2d = d3_2d.eval()
    #     o_d = d.eval()
    #     od3 = d3.eval()
    #     for j in range(3):
    #         for i in range(_n):
    #             lg.debug("checking %s" % repr((i,j)))
    #             assert np.allclose(od3[j * _n + i, ...], od3_2d[j, i, ...]), \
    #                 "No"
    #     for i in range(_n):
    #         lg.debug("\nvalues are\n%s, min is\n%s"
    #                  % (od3_2d[:, i, ...], o_d[i, ...]))

    if abs(d_padding) > 1e-3:
        d = tf.subtract(d, d_padding, name="joint_object_distances_padded")

    # mask inside
    mask_neg = tf.less(d, dtype_np(0.), 'jo_mask_neg')
    # lg.debug("mask: %s" % mask_neg)

    # mask same scenelet distances
    mask_same = tf.not_equal(
      tf.cast(pos_3d_transform_indices, tf.int64),
      tf.cast(obj_2d_poly_transform_indices[:, 0], tf.int64),
      name='jo_mask_same')
    if independent and um.has_static():
        mask_static_pose = tf.greater_equal(pos_3d_transform_indices,
                                            um.tid_static,
                                            name='mask_static_pose')
        mask_static_quad = tf.greater_equal(
          obj_2d_poly_transform_indices[:, 0], um.tid_static,
          name='mask_static_quad')

        xor = tf.logical_xor(
          mask_static_pose,
          mask_static_quad,
          name="static_dynamic_xor")
        mask_same = tf.logical_and(
          mask_same, xor, name='jo_mask_same_xor')
        lg.warning("TODO: check for static objects")

    mask = tf.logical_and(mask_neg, mask_same, name='jo_mask')
    # else:
    #     assert False
    #     mask = mask_neg

    # distance masked: (N, M)
    d = tf.where(mask, d, tf.zeros_like(d), name='jo_distances')

    #
    # loss
    #

    # per pose and quad
    loss_jo_unnorm_2d = tf.square(d, name='loss_jo_unn_2d')

    # per pose
    loss_jo_unnorm = tf.reduce_sum(input_tensor=loss_jo_unnorm_2d,
                                   axis=1, name='loss_jo_unnormalized')
    # if um.has_static():
    #     with tf.Session() as session:
    #         session.run(tf.global_variables_initializer())
    #         o = mask_same.eval()
    #         o2 = xor.eval()
    #         sys.exit(0)
    normalizer = tf.cast(x=joints.get_shape().as_list()[0], dtype=dtype_tf,
                         name='loss_jo_normalizer')
    loss_jo = tf.divide(loss_jo_unnorm,  # sqr. distances per joint
                        normalizer,
                        name='loss_jo_normalized')
    #
    # loss per transform
    #
    lg.warning("TODOOOO: add static poses to loss of quads' transforms.")
    pos_3d_transform_indices_1d = tf.squeeze(pos_3d_transform_indices, axis=-1)
    normalizer_per_transform = tf.segment_sum(
      data=tf.ones_like(pos_3d_transform_indices_1d, dtype=d.dtype),
      segment_ids=pos_3d_transform_indices_1d,
      name='normalizer_loss_jo_per_transform')

    loss_jo_per_transform = tf.Variable(
      initial_value=np.zeros(shape=(um.get_n_transforms), dtype=dtype_np))
    loss_jo_per_transform = tf.scatter_add(ref=loss_jo_per_transform,
                                           indices=pos_3d_transform_indices_1d,
                                           updates=loss_jo_unnorm,
                                           name='loss_jo_per_transform')
    loss_jo_per_transform = tf.divide(
      x=loss_jo_per_transform,
      y=normalizer_per_transform,
      name='loss_jo_per_transform_normalized')

    #
    # static-dynamic interactions
    #
    # e.g. a scenelet candidate's object is intersecting the initial path
    #  or an already placed scenelet's poses, then the cost of that should
    #  go to the object's scenelet and not to the already placed scenelet
    #  the poses belong to. (static pose - dynamic quad)
    if um.has_static() and independent:

        #
        # static poses' costs should be assigned to dynamic quads
        #

        mask_static_pose_dynamic_quad = tf.logical_and(
          x=xor,
          y=mask_static_pose,
          name='mask_static_pose_dynamic_quad')

        static_pose_costs_2d = tf.multiply(
          loss_jo_unnorm_2d,
          tf.cast(mask_static_pose_dynamic_quad, loss_jo_unnorm_2d.dtype),
          name='static_pose_costs_per_dyn_quad_2d')

        static_pose_costs = tf.reduce_sum(
          static_pose_costs_2d,
          axis=0,
          name='static_pose_costs_per_dyn_quad')

        static_pose_costs_per_transform_unn = tf.Variable(
          initial_value=np.zeros(shape=um.get_n_transforms),
          dtype=dtype_np)
        static_pose_costs_per_transform_unn = tf.scatter_add(
          ref=static_pose_costs_per_transform_unn,
          indices=obj_2d_poly_transform_indices[:, 0],
          updates=static_pose_costs,
          name='static_pose_costs_per_transform_unn')

        static_pose_normalizer_per_transform = tf.Variable(
          initial_value=np.zeros(shape=um.get_n_transforms),
          dtype=dtype_np)
        static_pose_normalizer_per_transform = tf.scatter_add(
          ref=static_pose_normalizer_per_transform,
          indices=obj_2d_poly_transform_indices,
          updates=tf.ones_like(obj_2d_poly_transform_indices, dtype=d.dtype),
          name='static_pose_costs_per_transform_normalizer')

        loss_jo_per_transform += tf.divide(
          static_pose_costs_per_transform_unn,
          static_pose_normalizer_per_transform,
          name='static_pose_costs_per_transform')

        # if um.has_static():
        #     with tf.Session() as session:
        #         session.run(tf.global_variables_initializer())
        #         o_mspdq = mask_static_pose_dynamic_quad.eval()
        #         o_static_pose_costs = static_pose_costs.eval()
        #         o = static_pose_costs_per_transform_unn.eval()
        #         o2 = loss_jo_per_transform.eval()
        #         sys.exit(0)

    return loss_jo, loss_jo_per_transform, \
           {'joints': joints, 'd': d}


def create_oo_intersection_losses(obj_2d_mgrid_vertices_transformed,
                                  obj_2d_polys,
                                  oo_mask_interacting,
                                  oo_mask_interacting_sum_inv):
    """Creates object-object losses.

    Args:

    Returns:
        losses

    """
    dtype_tf = obj_2d_polys.dtype
    dtype_np = dtype_tf.as_numpy_dtype

    with tf.name_scope('oo_intersection_loss'):
        d_oo = PointPolyDistanceEstimator.point_poly_distance(
          obj_2d_mgrid_vertices_transformed, obj_2d_polys, name='d_oo_raw')

    oo_mask_neg = tf.less(d_oo, dtype_np(0.), name='oo_mask_neg')
    oo_mask = tf.logical_and(oo_mask_interacting, oo_mask_neg, 'oo_mask')

    # TODO: angles

    d_oo = tf.where(condition=oo_mask, x=d_oo, y=tf.zeros_like(d_oo),
                    name='d_oo_masked')
    # normalize_oo = tf.reduce_sum(tf.cast(oo_mask_interacting,
    # dtype=tf.int32))
    # lg.debug("normalize_oo: %s" % normalize_oo)
    loss_oo_unn = tf.nn.l2_loss(d_oo, name='loss_oo_unn')
    if oo_mask_interacting_sum_inv is not None:
        loss_oo = tf.multiply(
          x=loss_oo_unn,
          y=oo_mask_interacting_sum_inv,
          name='loss_oo')
    else:
        loss_oo = loss_oo_unn

    # TODO: loss_oo_per_transform
    loss_oo_per_transform = None

    # deb = locals()
    return loss_oo, loss_oo_per_transform, \
           {'d_oo': d_oo, 'oo_mask': oo_mask}
