import numpy as np
import tensorflow as tf


class PairedPointPolyDistanceEstimator(object):

    @staticmethod
    def point_poly_distance(p, poly, is_wrapped,
                            point_coords=(0, 2), poly_coords=(0, 2),
                            do_checks=True, name=None):
        """Part of TF graph that computes point to polygon distance.

        Arguments:
            p (N, 3):
                Points in rows, 2D or 3D.
            poly (N, NV, 3):
                Polygons in first dim, with 2D or 3D points in last dim.a
            is_wrapped (bool):
                Is the input poly wrapped (True) or needs wrapping (False).
            point_coords (tuple):
                Coordinates to use for top-view from p.
            poly_coords (tuple):
                Coordinates to use for top-view from poly.
            name (str):
                TensorFlow operation name to set the output distances to.
        Returns:
            An operation of size (N) containing the signed distances of
            point p_i and polygon poly_i, i=1..N
        """
        if do_checks:
            assert len(point_coords) == 2
            assert len(poly_coords) == 2
            shape_p = p.get_shape().as_list()
            assert len(shape_p) == 2, \
                "Need points in rows. %s" % repr(shape_p)
            assert shape_p[1] in (2, 3), "Need x,y,z or x,z for points."
            shape_poly = poly.get_shape().as_list()
            assert len(shape_poly) == 3, \
                "Need polygon points as N x NV x (2, 3)"
            assert shape_poly[2] in (2, 3), "Assumed poly points in 2D or 3D."
        dtype_tf = p.dtype
        dtype_np = dtype_tf.as_numpy_dtype

        # wrap, if needed
        if not is_wrapped:
            poly = tf.concat((poly, poly[:, 0:1, :]), axis=1)

        x = tf.slice(p, [0, point_coords[0]], [-1, 1])  # column vector
        y = tf.slice(p, [0, point_coords[1]], [-1, 1])  # column vector

        # edge vertices
        xv0 = poly[:, :-1, poly_coords[0]]
        xv1 = poly[:, 1:, poly_coords[0]]
        yv0 = poly[:, :-1, poly_coords[1]]
        yv1 = poly[:, 1:, poly_coords[1]]

        # parameters of the lines co-linear with each polygon edge:
        #  (A,B) is normal, C is distance from origin projected onto normal
        A = yv0 - yv1
        B = xv1 - xv0
        C = yv1 * xv0 - xv1 * yv0

        # find projection (xp,yp) of points onto lines co-linear with edges
        denom = tf.square(A) + tf.square(B)
        # AB_ = tf.divide(1., tf.where(tf.abs(denom) > 1.e-6, denom,
        #                 np.float64(1.e9) * tf.ones_like(denom)))
        AB_ = tf.divide(dtype_np(1.),
                        tf.where(tf.not_equal(denom, dtype_np(0.)),
                                 denom,
                                 dtype_np(1.e9) * tf.ones_like(denom)),
                        name='AB_')
        Ax = A * x
        By = B * y
        vv = Ax + By + C
        xp = x - A * AB_ * vv
        yp = y - B * AB_ * vv

        # handle horizontal and vertical edges (either xp or yp is unstable
        # for these edges, replace it with an edge vertex coordinate)
        xp = tf.where(B == dtype_np(0.), xv0, xp)
        yp = tf.where(A == dtype_np(0.), yv0, yp)

        # find points with projections inside edges
        xp_inside_edge = tf.logical_or(
            tf.logical_and(
                tf.greater_equal(xp, xv0),
                tf.less_equal(xp, xv1)
            ),
            tf.logical_and(
                tf.greater_equal(xp, xv1),
                tf.less_equal(xp, xv0)
            )
        )
        yp_inside_edge = tf.logical_or(
            tf.logical_and(
                tf.greater_equal(yp, yv0),
                tf.less_equal(yp, yv1)
            ),
            tf.logical_and(
                tf.greater_equal(yp, yv1),
                tf.less_equal(yp, yv0)
            )
        )
        inside_edge = tf.logical_and(xp_inside_edge, yp_inside_edge)

        # squared distance to polygon vertices
        dv_sq = tf.square(xv0 - x) + tf.square(yv0 - y)

        # squared distance to lines co-linear with edges
        dp_sq = tf.square(xp - x) + tf.square(yp - y)

        # squared distance to polygon is the minimum over sq. distance to
        # vertices and sq. distance to projected points inside edges
        # also, distance to vertices is always >= distance to adjacent edges,
        # so it is safe to use projected points if they
        # are inside an edge instead of adjacent vertices
        d_sq = tf.reduce_min(tf.where(inside_edge, dp_sq, dv_sq), axis=-1)
        small_fill = tf.ones_like(d_sq) * dtype_np(1.e-3)
        d = tf.sqrt(  # avoid undefined gradient at 0
          tf.where(tf.not_equal(d_sq, dtype_np(0.)), d_sq, small_fill))

        # signed 0-crossings of winding as inside test
        # see http://geomalgorithms.com/a03-_inclusion.html#wn_PnPoly()

        # > 0 if left of edge line, = 0 if on edge line,
        # < 0 if right of edge line
        is_left = (y - yv0) * B - (x - xv0) * -A
        winding = tf.where(
            condition=tf.less_equal(yv0, y),
            x=tf.cast(tf.logical_and(tf.greater(is_left, dtype_np(0.)),
                                     tf.greater(yv1, y)), dtype_tf),
            y=-tf.cast(tf.logical_and(tf.less(is_left, dtype_np(0.)),
                                      tf.less_equal(yv1, y)), dtype_tf),
            name='winding')
        # inside only if the winding number is 0
        inside = tf.not_equal(tf.reduce_sum(winding, axis=-1), 0)

        d = tf.where(inside, -d, d, name=name)

        # return [is_left, winding, inside, d, B, A]

        return d

    @staticmethod
    def test(p_in, poly_in, point_coords=(0, 2), poly_coords=(0, 2)):

        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:0'):
            p = tf.Variable(initial_value=p_in, dtype=tf.float64,
                            trainable=False)
            poly = tf.Variable(initial_value=poly_in, dtype=tf.float64,
                               trainable=False)

            d = PairedPointPolyDistanceEstimator.point_poly_distance(p, poly, point_coords, poly_coords)

        with tf.Session(graph=graph) as session:
            # init
            session.run(tf.global_variables_initializer())

            d_out = session.run(d)

        return d_out

if __name__ == '__main__':
    import scipy.io

    test_data_filepath = '../../../../data/paired_point_poly_distance_test.mat'
    test_data = scipy.io.loadmat(test_data_filepath)

    py_p = test_data['p_list']
    py_poly = test_data['poly_list']
    py_d_gt = test_data['d_list'].squeeze()

    py_d = PairedPointPolyDistanceEstimator.test(py_p, py_poly, point_coords=(0, 1), poly_coords=(0, 1))

    max_diff = np.max(np.abs(py_d - py_d_gt))

    print('max difference: %e' % max_diff)

    assert np.allclose(py_d, py_d_gt, atol=0.00001)
