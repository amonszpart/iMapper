import os
import sys
import numpy as np
import tensorflow as tf

from imapper.pose.paired_point_poly_distance_estimator \
    import PairedPointPolyDistanceEstimator


class PairedOcclusionV1(object):

    @staticmethod
    def point_quad_occlusion_distance(p, quads,
                                      name='point_quad_occlusion_distance',
                                      z_scale=20., backface_culling=True):
        """Part of TF graph that computes point to occlusion volume
        signed distance. The camera is assumed to be at (0, 0, 0).

        Arguments:
            p (N, 3):
                Points in rows, 2D or 3D.
            quads (N, NV, 3):
                quads in first dim, with 3D points in last dim.
            name (str):
                Name of output operation
            z_scale (float):
                How far away the occlusion volume rectangles should stretch.
            backface_culling (bool):
                Saves half of computations by using scatter_nd, that causes
                a large sparse-to-dense conversion in the gradient computation.
        Returns:
            An operation of size (N) containing the signed distances of
            point p_i and polygon poly_i, i=1..N
        """
        shape_p = p.get_shape().as_list()
        assert len(shape_p) == 2, "Need points in rows. %s" % repr(shape_p)
        assert shape_p[1] == 3, "Need x,y,z for points."
        shape_poly = quads.get_shape().as_list()
        assert len(shape_poly) == 3, "Need polygon points as N x NV x 3"
        assert shape_poly[2] == 3, "Assumed poly points in 3D."
        assert shape_p[0] == shape_poly[0], "Assumed pairs of points and quads"

        # wrap: copy first corner to last
        wrapped = tf.concat((quads, quads[:, 0:1, :]), axis=1,
                            name='quads_wrapped')

        # we need these new rectangles (columns 2 and 3 will be scaled)
        # r1: 0, 1, 1, 0, 0
        # r2: 1, 2, 2, 1, 1
        # r3: 2, 3, 3, 2, 2
        # r4: 3, 4, 4, 3, 3

        # transposed, they are the corners in order
        # r1, r2, r3, r4
        #  0,  1,  2,  3 = wrapped[:, :-1, :]
        #  1,  2,  3,  4 = wrapped[:, 1:, :]
        #  1,  2,  3,  4 = wrapped[:, 1:, :]
        #  0,  1,  2,  3 = wrapped[:, :-1, :]
        #  0,  1,  2,  3 = wrapped[:, :-1, :]

        # extract columns once
        cols_03 = wrapped[:, :-1, :]
        cols_12 = wrapped[:, 1:, :]

        n = p.shape[0]

        # 5 quads: original, and 4 sides
        # shape is (N, 5, 5, 3), where
        #  the 0th index is the original polygon id
        #  the first index is the side-polygon id
        #  the second index is the corner id and
        #  the third index is x, y, z
        side_polys_g = tf.concat((
            wrapped[:, None, :, :],     # prepend original wrapped
            tf.stack((cols_03, cols_12, cols_12 * z_scale, cols_03 * z_scale,
                      cols_03),         # wrap around
                     axis=2)),          # axis=1 would be non-transposed
            axis=1, name='side_polys')  # prepend original polygon
        # lg.debug("side_polys: %s" % side_polys)

        # compute normals of all quads
        # quad edge vectors -> (N,5,5,3)
        edges = tf.subtract(side_polys_g[:, :, 1:, :],
                            side_polys_g[:, :, :-1, :], name='edges')
        # cross product of adjacent edges (unnorm. normal for each triangle spanned by adjacent edges) -> (N,5,5,3)
        normals = tf.cross(edges[:, :, :-1, :], edges[:, :, 1:, :], name='triangle_normals')
        # average of all cross products -> (N,5,1,3)
        # this only works for quads! This would fail for 5-gons and the normalized edge version would fail for 6-gons (maybe less)
        normals = tf.reduce_mean(normals, axis=2, keepdims=True, name='unit_normals')
        # normalize normal -> (N,5,1,3)
        normals /= tf.norm(normals, axis=3, keepdims=True)

        # get axes of local coordinate frame of each quad
        # (where the quad is in the xz plane and the normal points towards y) -> (N,5,1,3)
        localx = edges[:, :, 0:1, :]
        localx /= tf.norm(localx, axis=3, keepdims=True)
        localz = tf.cross(normals, localx) # no need to normalize, localx and normals have to be orthogonal and unit length

        # get location of points and vertices in local coordinate frames of quads -> (N,5,3,3)
        rotmat = tf.concat((localx, normals, localz), axis=2)

        # (N,5,3,3), (N,5,5,3) -> (N,5,5,3) (N,polys,verts,dims)
        side_polys = tf.einsum('ijlm,ijkm->ijkl', rotmat, side_polys_g)

        # (N,5,3,3), (N,3) -> (N,5,3) (N,polys,dims)
        p = tf.einsum('ijlm,im->ijl', rotmat, p)

        # remove quad 5-tuples where the original quad is not facing the camera
        # (those have negative y now) N -> N'
        facing = tf.less(side_polys[:, 0, 0, 1], 0.)
        if backface_culling:
            facing_inds = tf.squeeze(tf.where(facing))
            side_polys = tf.gather(side_polys, indices=facing_inds, axis=0)
            p = tf.gather(p, indices=facing_inds, axis=0)

        # height above the quads (negative means behind) -> (N',5)
        height = p[:, :, 1] - side_polys[:, :, 0, 1]

        # screen position of the point on the plane of the original quad -> (N',3)
        # point far away if the plane height is nearly 0 (plane is almost parallel to viewing direction)
        p_screen = tf.where(
            tf.abs(p[:, 0, 1]) >= 0.0001,
            p[:, 0, :] * tf.expand_dims(1 - height[:, 0] / p[:, 0, 1], axis=-1),
            tf.ones_like(p[:, 0, :]) * 99999)

        # distance of screen point to original quad in the xz plane -> (N')
        d_screen = PairedPointPolyDistanceEstimator.point_poly_distance(
            p_screen, side_polys[:, 0, :, :], is_wrapped=True,
            point_coords=(0, 2),
            poly_coords=(0, 2),
            do_checks=False)

        # reshape to single list over all quads
        # -> (N'*5,3)
        p = tf.reshape(p, shape=[-1, 3])
        # -> (N'*5,5,3)
        side_polys = tf.reshape(side_polys, shape=[-1, 5, 3])

        # distance to quads in the xz plane -> (N'*5)
        d = PairedPointPolyDistanceEstimator.point_poly_distance(
            p, side_polys, is_wrapped=True,
            point_coords=(0, 2),
            poly_coords=(0, 2),
            do_checks=False)

        # reshape to list over all original quads x 5
        # -> (N',5)
        d = tf.reshape(d, [-1, 5])

        # point is inside the occlusion volume if it is inside the original quad in the quad plane
        # and behind the plane of the original quad
        # -> (N',5)
        inside = tf.logical_and(
            tf.less(d_screen, 0.), # inside original quad in screen space
            tf.less(height[:, 0], 0.)) # behind original quad

        # distance to the quads in 3d -> (N',5)
        d = tf.sqrt(tf.where(
            tf.less(d, 0.),
            tf.square(height),
            tf.square(height) + tf.square(d)))

        # smallest distance over the 5 quads -> (N')
        d = tf.reduce_min(d, axis=-1)

        # distance to the occlusion volume -> (N')
        d = tf.where(inside, -d, d)

        # back to all quads (not only front-facing) N' -> N, back-facing quads get 99999 distance to occlusion volume
        if backface_culling:
            d = tf.scatter_nd(indices=tf.expand_dims(facing_inds, -1), updates=d, shape=[n])
        d = tf.where(facing, d, tf.ones_like(d)*9999., name=name)
            # d = tf.scatter_update(d, indices=facing_away_inds, updates=tf.constant([9999]))

        # return [facing_inds, height, p_screen, p, side_polys, d_screen, inside, d_facing, d]

        return d

    @staticmethod
    def test(p_in, quads_in, z_scale=20.):

        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:0'):
            p = tf.Variable(initial_value=p_in, dtype=tf.float32, trainable=False)
            quads = tf.Variable(initial_value=quads_in, dtype=tf.float32, trainable=False)

            d = PairedOcclusionV1.point_quad_occlusion_distance(p=p, quads=quads, z_scale=z_scale)

        with tf.Session(graph=graph) as session:
            # init
            session.run(tf.global_variables_initializer())

            d_out = session.run(d)

        return d_out


if __name__ == '__main__':
    # import numpy as np

    # load test data
    p_in = np.load('../../../../data/points.npy')
    quads_in = np.load('../../../../data/polygons.npy')
    # d_gt = np.load('../../../../data/distances_gt.npy')

    # run test
    x = PairedOcclusionV1.test(p_in, quads_in)

    np.save('../../../../data/distances_gt.npy', x)

    # import pdb; pdb.set_trace()

    # # compare result to gt
    # max_diff = np.max(np.abs(d_out - d_out_gt))

    # print('max difference: %e' % max_diff)

    # assert np.allclose(d_out, d_out_gt, atol=0.00001)
