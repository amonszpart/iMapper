import numpy as np
import tensorflow as tf
from imapper.util.stealth_logging import lg
from imapper.logic.categories import CATEGORIES


class TFVariablesManager(object):
    """Keeps track of TensorFlow variables in the optimization."""

    def __init__(self):

        #
        # 2D
        #

        self._obj_2d_vertices = None
        """(N * 4, 3) bounding rectangle corners, y component is always 0. 
        formerly: obj_vxs"""

        self._obj_2d_transform_indices = None
        """(N, 1) Former: transform_indices"""

        self._obj_2d_transforms_tiled = None
        """(N * 4, 3, 4) One 3D transform for each vertex."""

        self._obj_2d_vertices_transformed = None
        """(N * 4, 3) transformed rectangle corners."""

        self._obj_2d_polys = None
        """(N, 4, 3) transformed top-view rectangles."""

        self._obj_2d_poly_transform_indices = None
        """(N, 4) transform indices of top-view rectangles. Has same value in 
        each row.
        """

        self._obj_2d_angles_per_poly = None
        """(N,) angle of object forward when it was imported from disk."""

        self._obj_2d_mgrid_vertices_transformed = None

        self._oo_mask_interacting = None

        self._np_oo_sum = None
        """non-inverse normalizer on the CPU"""

        self._oo_mask_interacting_sum_inv = None
        """loss_oo normalizer"""

        self.oo_mask_cat = None
        
        self.cat_ids_polys = None
        """Category ids for each polygon."""

        #
        # 3D
        #

        self._obj_3d_vxs = None
        """(M * 6 * 4, 3) object part OBB vertices."""

        self._obj_3d_transform_indices = None
        """(M * 6 * 4, 1) Indices for each corner of each polygon of 
        each object part.
        """

        self._obj_3d_vertices_transformed = None
        """(M * 6 * 4, 3) Transformed part OBB vertices."""

        self._obj_3d_polys = None
        """(M * 6, 4, 3) Transformed part OBB side quads."""

        self._obj_3d_poly_transform_indices = None
        """(M * 6, 4) poly transform ids. Each row is same."""

    def create_objects_2d(self, transforms, um):
        """Creates variables for intersection terms."""

        # rectangle corners
        self._obj_2d_vertices = tf.Variable(
          initial_value=um.obj_2d_vertices, trainable=False,
          dtype=transforms.dtype, name='obj_2d_vertices')

        # transformation indices for each object vertex
        self._obj_2d_transform_indices = tf.constant(
          value=um.obj_2d_transform_indices[:, None],
          shape=[len(um.obj_2d_transform_indices), 1],
          name='obj_2d_transform_indices', dtype=tf.int32)

        # transformations for object vertices
        self._obj_2d_transforms_tiled = tf.gather_nd(
          params=transforms,
          indices=self._obj_2d_transform_indices,
          name="obj_2d_transforms_tiled")

        # transform object vertices
        self._obj_2d_vertices_transformed = tf.add(
          x=tf.squeeze(
            input=tf.matmul(a=self._obj_2d_transforms_tiled[:, :, :3],
                            b=self._obj_2d_vertices[:, :, None]),
            axis=-1),
          y=self._obj_2d_transforms_tiled[:, :, 3],
          name='obj_2d_vertices_transformed'
        )

        # reshape to polygons
        self._obj_2d_polys = tf.reshape(
          tensor=self._obj_2d_vertices_transformed,
          shape=(-1, 4, 3),
          name='obj_2d_polys')

        # polygon transform indices
        self._obj_2d_poly_transform_indices = tf.reshape(
          tensor=self._obj_2d_transform_indices,
          shape=(-1, 4),
          name='obj_2d_poly_indices')

        # original polygon orientations (for spatial arrangement term)
        self._obj_2d_angles_per_poly = tf.Variable(
          initial_value=um.obj_2d_angles_per_poly,
          trainable=False,
          dtype=transforms.dtype,
          name='obj_2d_angles_per_poly')
        
        # 1 x n_polys
        self.cat_ids_polys = tf.constant(
            value=um.obj_2d_cat_ids_per_poly[None, :],
            name='cat_ids_polys', dtype=tf.int32)

    def create_objects_2d_mgrids(self, transforms, um):
        assert self.is_objects_2d_created, "Call create_objects_2d first"
        dtype_tf = transforms.dtype
        dtype_np = dtype_tf.as_numpy_dtype

        mgrid_vxs = tf.Variable(initial_value=um.obj_2d_mgrid_vxs,
                                trainable=False,
                                name='obj_2d_mgrid_vertices')
        mgrid_indices = tf.constant(
          value=um.obj_2d_mgrid_transform_indices[:, None],
          name='mgrid_indices', dtype=tf.int32)
        mgrid_transforms_tiled = tf.gather_nd(transforms,
                                              indices=mgrid_indices,
                                              name='mgrid_transforms_tiled')

        self._obj_2d_mgrid_vertices_transformed = tf.add(
          x=tf.squeeze(tf.matmul(a=mgrid_transforms_tiled[:, :, :3],
                                 b=mgrid_vxs[:, :, None]),
                       axis=-1),
          y=mgrid_transforms_tiled[:, :, 3],
          name="obj_2d_mgrid_vertices_transformed")

        self.oo_mask_same = tf.not_equal(
          tf.cast(mgrid_indices, dtype=tf.int64),
          tf.cast(tf.transpose(self._obj_2d_poly_transform_indices[:, 0:1]),
                  dtype=tf.int64),
          name='oo_mask_same')

        lg.warning("Use numpy based mask for less memory")
        # TODO: use this instead
        oo_mask_same_np = np.not_equal(
            um.obj_2d_mgrid_transform_indices[:, None],
            um.obj_2d_transform_indices[None, ::4]).astype('b1')

        # self.oo_mask_same_2 = tf.constant(
        #   value=oo_mask_same_np,
        #   dtype=tf.bool,
        #   shape=(mgrid_indices.get_shape().as_list()[0],
        #          self._obj_2d_poly_transform_indices.get_shape().as_list()[0]),
        #   name='oo_mask_2',
        #   verify_shape=True)

        # n_grid_points x 1
        lg.warning("Use numpy based cat mask for less memory")
        cat_ids_mgrids = tf.constant(value=um.obj_2d_mgrid_cat_ids[:, None],
                                     name='oo_cat_ids_mgrids', dtype=tf.int32)
        # n_grid_points x n_polys
        self.oo_mask_cat = tf.logical_and(
          tf.not_equal(tf.cast(cat_ids_mgrids, tf.int64),
                       tf.cast(self.cat_ids_polys, tf.int64),
                       name='oo_mask_cat_same'),
          tf.constant(um.obj_2d_cat_ids_per_poly[None, :]
                      != CATEGORIES['table'],
                      dtype=tf.bool, name='oo_mask_cat_is_table'),
          name='oo_mask_cat')

        oo_mask_cat_np = np.not_equal(
            um.obj_2d_mgrid_cat_ids[:, None],
            um.obj_2d_cat_ids_per_poly[None, :]).astype('b1')
        self.oo_mask_cat_2 = tf.constant(
          oo_mask_cat_np,
          shape=(cat_ids_mgrids.get_shape().as_list()[0],
                 self.cat_ids_polys.get_shape().as_list()[1]),
          verify_shape=True,
          name='oo_mask_cat_2'
        )

        # TODO: use this instead
        self.oo_mask_interacting_2 = np.logical_and(oo_mask_same_np,
                                                    oo_mask_cat_np).astype('b1')
        self._oo_mask_interacting = tf.logical_and(self.oo_mask_cat,
                                                   self.oo_mask_same,
                                                   name='oo_mask_interacting')

        # self._oo_mask_interacting_sum_inv = tf.reciprocal(
        #   tf.cast(tf.reduce_sum(tf.cast(self._oo_mask_interacting,
        #                                 dtype=tf.int32)),
        #           dtype=dtype_tf))

        # TODO: use this instead
        self._np_oo_sum = np.sum(self.oo_mask_interacting_2.astype('i4')) \
            .astype(dtype_np)
        self._oo_mask_interacting_sum_inv = tf.constant(
          value=np.reciprocal(self._np_oo_sum if self._np_oo_sum > 0 else 1),
          dtype=dtype_tf, name='loss_oo_normalizer')

        # with tf.Session() as session:
        #     lg.info("[match] variables_initializer")
        #     # Init
        #     session.run(tf.global_variables_initializer())
        #     o = self.oo_mask_cat.eval()
        #     o_2 = self._oo_mask_interacting.eval()
        #     sys.exit(0)

    def create_objects_3d(self, um, transforms):
        """Creates 3D TensorFlow variables for the 3D OBBs.

        Returns:
            _obj_3d_polys (tf.Variable): (M * 6, 4, 3)
            _obj_3d_poly_transforms_indices (tf.Variable): (M * 6, 4)
        """

        self._obj_3d_vxs = tf.Variable(initial_value=um.object_vertices,
                                       trainable=False,
                                       name='obj_3d_vertices')

        # transformation indices for each object vertex
        self._obj_3d_transform_indices = tf.constant(
          value=um.object_transform_indices[:, None],
          shape=[len(um.object_transform_indices), 1],
          name='obj_3d_transform_indices')

        # transformations for object vertices
        transforms_tiled = tf.gather_nd(params=transforms,
                                        indices=self._obj_3d_transform_indices,
                                        name="obj_3d_transforms_tiled")

        # transform
        self._obj_3d_vertices_transformed = tf.add(
          x=tf.squeeze(tf.matmul(transforms_tiled[:, :, :3],
                                 self._obj_3d_vxs[:, :, None]), axis=-1),
          y=transforms_tiled[:, :, 3],
          name="obj_3d_vertices_transformed")

        # reshape vertices to polygons, each consecutive 6 forms and OBB
        self._obj_3d_polys = tf.reshape(
          tensor=self._obj_3d_vertices_transformed,
          shape=(-1, 4, 3),
          name="obj_3d_polys")

        self._obj_3d_poly_transform_indices = tf.reshape(
          tensor=self._obj_3d_transform_indices,
          shape=(-1, 4),
          name="obj_3d_poly_transform_indices")

        return self._obj_3d_polys, self._obj_3d_poly_transform_indices

    @property
    def is_objects_2d_created(self):
        return self._obj_2d_polys is not None

    @property
    def obj_2d_polys(self):
        assert self._obj_2d_polys is not None, "Call create_objects_2d first"
        return self._obj_2d_polys

    @property
    def obj_2d_vertices_transformed(self):
        assert self._obj_2d_polys is not None, "Call create_objects_2d first"
        return self._obj_2d_vertices_transformed

    @property
    def obj_2d_transform_indices(self):
        assert self._obj_2d_poly_transform_indices is not None, \
            "Call create_objects_2d first"
        return self._obj_2d_poly_transform_indices

    @property
    def obj_2d_mgrid_vertices_transformed(self):
        return self._obj_2d_mgrid_vertices_transformed

    @property
    def oo_mask_interacting(self):
        return self._oo_mask_interacting

    @property
    def oo_mask_interacting_sum_inv(self):
        return self._oo_mask_interacting_sum_inv

    @property
    def obj_3d_poly_transform_indices(self):
        return self._obj_3d_poly_transform_indices
