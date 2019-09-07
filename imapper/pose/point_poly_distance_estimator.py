import json

import numpy as np
import tensorflow as tf

from imapper.util.stealth_logging import lg


class PointPolyDistanceEstimator(object):

    @staticmethod
    def point_poly_distance(p, poly,
                            point_coords=(0, 2), poly_coords=(0, 2),
                            name=None):
        """Part of TF graph that computes point to polygon distance.

        Arguments:
            p: (N, 3)
                Points in rows, 2D or 3D.
            poly: (M, 4, 3)
                Polygons in first dim, with 2D or 3D points in last dim.a
            point_coords: (tuple)
                Coordinates to use for top-view from p.
            poly_coords: (tuple)
                Coordinates to use for top-view from poly.
            name (str):
                TensorFlow operation name.
        Returns:
            An operation of size (N, M) containing the signed distance of
            point p_i, i <= N and polygon poly_j, j <= M.
        """
        dtype_tf = p.dtype
        dtype_np = dtype_tf.as_numpy_dtype

        assert len(point_coords) == 2
        assert len(poly_coords) == 2
        shape_p = p.get_shape().as_list()
        assert len(shape_p) == 2, "Need points in rows. %s" % repr(shape_p)
        assert shape_p[1] in (2, 3), "Need x,y,z or x,z for points."
        shape_poly = poly.get_shape().as_list()
        assert len(shape_poly) == 3, "Need polygon points as M x 4 x (2, 3)"
        assert shape_poly[2] in (2, 3), "Assumed poly points in 2D or 3D."
        poly = tf.concat((poly, poly[:, 0:1, :]), axis=1)
        # p = tf.Variable(initial_value=py_p[:, [0, 2]], dtype=tf.float32,
        #                 trainable=False)
        # poly = tf.Variable(initial_value=wrapped, dtype=tf.float32,
        #                    trainable=False)

        x = p[:, point_coords[0]]
        y = p[:, point_coords[1]]

        xv0 = poly[:, :-1, poly_coords[0]]
        xv1 = poly[:, 1:, poly_coords[0]]
        yv0 = poly[:, :-1, poly_coords[1]]
        yv1 = poly[:, 1:, poly_coords[1]]

        A = yv0 - yv1
        B = xv1 - xv0
        C = yv1 * xv0 - xv1 * yv0
        denom = tf.square(A) + tf.square(B)
        # sel = tf.abs(denom) > 1.e-6
        sel = tf.not_equal(denom, tf.cast(0., dtype_tf))
        AB_ = tf.divide(
          dtype_np(1.),
          tf.where(sel, denom,
                   dtype_np(1.e9) * tf.ones_like(denom)
                   )
        )
        Ax = tf.einsum('ai,b->bai', A, x)
        By = tf.einsum('ai,b->bai', B, y)
        vv = Ax + By + C
        xbrd = x[:, None, None]
        xp = xbrd - tf.einsum('abi,bi->abi', vv, (A * AB_))
        ybrd = y[:, None, None]
        yp = ybrd - tf.einsum('abi,bi->abi', vv, (B * AB_))
        xp = tf.where(B == 0.,
                      tf.tile(xv0[None, :, :], (tf.shape(p)[0], 1, 1)),
                      xp)
        yp = tf.where(A == 0.,
                      tf.tile(yv0[None, :, :],
                              (tf.shape(p)[0], 1, 1)),
                      yp)

        idx_x = tf.logical_or(
            tf.logical_and(
                tf.greater_equal(xp, xv0),
                tf.less_equal(xp, xv1)
            ),
            tf.logical_and(
                tf.greater_equal(xp, xv1),
                tf.less_equal(xp, xv0)
            )
        )
        idx_y = tf.logical_or(
            tf.logical_and(
                tf.greater_equal(yp, yv0),
                tf.less_equal(yp, yv1)
            ),
            tf.logical_and(
                tf.greater_equal(yp, yv1),
                tf.less_equal(yp, yv0)
            )
        )
        idx = tf.logical_and(idx_x, idx_y)

        dv_sum = tf.square(xv0 - xbrd) + tf.square(yv0 - ybrd)
        # small_fill = tf.fill(dv_sum.get_shape(), np.float64(1.e-3))
        small_fill = tf.ones_like(dv_sum) * dtype_np(1.e-3)
        dv = tf.sqrt(
          tf.where(tf.not_equal(dv_sum, dtype_np(0.)), dv_sum, small_fill)
        )
        # dv = tf.sqrt(dv_sum)
        shp_ = idx.get_shape().as_list()
        dp_sum = tf.square(xp - xbrd) + tf.square(yp - ybrd)
        dp_sqrt = tf.sqrt(
          tf.where(tf.not_equal(dp_sum, dtype_np(0.)), dp_sum, small_fill)
        )
        dp = tf.where(idx,
                      dp_sqrt,
                      # tf.fill(shp_, tf.float64.max)
                      tf.ones_like(idx, dtype=dtype_tf) * dtype_tf.max
                      )
        min_dp = tf.reduce_min(dp, axis=-1)
        min_dv = tf.reduce_min(dv, axis=-1)
        d = tf.where(tf.reduce_any(idx, axis=-1),
                     tf.minimum(min_dp, min_dv),
                     min_dv)
        if False:
            dots = (xbrd - xv0) * B + (ybrd - yv0) * -A
            inside = tf.reduce_all(tf.greater_equal(x=dots, y=dtype_np(0.)),
                                   axis=-1)
        else:
            # signed 0-crossings of winding as inside test
            # see http://geomalgorithms.com/a03-_inclusion.html#wn_PnPoly()

            # > 0 if left of edge line, = 0 if on edge line,
            # < 0 if right of edge line
            xv0_brd = xv0[None, :, :]
            yv0_brd = yv0[None, :, :]
            is_left = (ybrd - yv0_brd) * B[None, :, :] \
                - (xbrd - xv0_brd) * -A[None, :, :]
            winding = tf.where(
              condition=tf.less_equal(yv0_brd, ybrd),
              x=tf.cast(tf.logical_and(tf.greater(is_left, dtype_np(0.)),
                                       tf.greater(yv1[None, :, :], ybrd)),
                        dtype=tf.int32),
              y=-tf.cast(tf.logical_and(tf.less(is_left, dtype_np(0.)),
                                        tf.less_equal(yv1[None, :, :], ybrd)),
                         dtype=tf.int32),
              name='winding')

            # inside only if the winding number is 0
            # reduce_sum only works on int32
            sum_winding = tf.reduce_sum(winding, axis=-1, name='sum_winding')
            # not_equal only works on int64 out of the two ints
            inside = tf.not_equal(tf.cast(sum_winding, tf.int64),
                                  0, name='winding_is_zero')

        d = tf.where(inside, -d, d, name=name)

        return d

    @staticmethod
    def test(py_p, py_poly):
        np.set_printoptions(suppress=True)
        with open('/tmp/test_p_poly_dist.json', 'r') as fin:
            jdata = json.load(fin)

        wrapped = []
        for row in py_poly:
            wrapped.append(np.concatenate((row, row[0:1, :])))
        wrapped = np.array(wrapped, dtype=np.float64)
        lg.debug("wrapped:\n%s" % wrapped)
        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:0'):
            p = tf.Variable(initial_value=py_p, dtype=tf.float64,
                            trainable=False)
            poly = tf.Variable(initial_value=wrapped, dtype=tf.float64,
                               trainable=False)
            # d, g, sel = PointPolyDistanceEstimator.point_poly_distance(p, poly)
            d = PointPolyDistanceEstimator.point_poly_distance(p, poly)

        with tf.Session(graph=graph) as session:
            # init
            session.run(tf.global_variables_initializer())

            o_d = session.run(d)
            # o_d, o_g, o_sel = session.run([d, g, sel])
            # lg.debug("gradients: %s" % o_g)
            # lg.debug("sel: %s" % o_sel)

        lg.debug("o_d (%s):\n%s" % (repr(o_d.shape), o_d))
        assert np.allclose(o_d,
                           np.squeeze(np.array(jdata['d']), axis=-1), atol=0.001), \
            "no:\n%s\n%s\n%s\n%s" \
            % ('o_d: ', o_d,
               'gt d: ', np.squeeze(np.array(jdata['d']), axis=-1))