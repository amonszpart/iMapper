import codecs
import json
import os
import warnings

import numpy as np

try:
    from numba import jit
except ImportError:
    print("Could not import jit from numba...")
#
    def jit(cache=None, nopython=None):
        def true_decorator(f):
            return f
        return true_decorator

from imapper.logic import geometry
from imapper.logic.pointTriangleDistance import pointTriangleDistance
from imapper.scenelet_fit.consts import TWO_PI

import sys
if not sys.version_info[0] < 3:
    from functools import lru_cache
else:
    from repoze.lru import lru_cache
from imapper.logic.scene_object_interface import SceneObjectInterface


# @profile
#"b1(f4[:], f4[:, :], f4[:],f4[:], f4[:, :], f4[:])"
# @jit(nopython=True)
@jit(cache=True, nopython=True)
def _intersects(c0, A, scales0, c1, B, scales1):
    """
    Source: https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf
    :param other: Other obb to intersect
    :return: 
    """
    # A = gm.normalized(self.axes, axis=0)
    EA = scales0 / np.float32(2.)
    # assert EA.dtype == np.float32, "Wrong type: %s" % EA.dtype

    # B = gm.normalized(other.axes, axis=0)
    # B = other._axes
    EB = scales1 / np.float32(2.)
    # assert EB.dtype == np.float32, "Wrong type: %s" % EB.dtype

    # C = np.matmul(A.T, B)  # c_ij in pdf
    C = A.T * B  # c_ij in pdf
    # assert C.dtype == np.float32, "wrong type: %s" % C.dtype
    absC = np.abs(C)

    # // Compute the translation vector.
    D = c1 - c0  # D in pdf

    # AD = np.matmul(A.T, D)
    # print(repr(A.T.shape))
    # print(repr(D.shape))
    # AD2 = np.array([
    #     A[0, 0] * D[0] + A[1, 0] * D[1] + A[2, 0] * D[2],
    #     A[0, 1] * D[0] + A[1, 1] * D[1] + A[2, 1] * D[2],
    #     A[0, 2] * D[0] + A[1, 2] * D[1] + A[2, 2] * D[2]])

    # print("dtypec0: ", c0.dtype)
    # print("dtypec1: ", c1.dtype)
    # print("dtypeA: ", A.dtype)
    # print("dtypeD: ", D.dtype)
    AD = np.dot(A.T, D)
    # assert AD.dtype == np.float32, "wrong type: %s" % AD.dtype
    # assert np.allclose(AD, AD2), "Not close: %s %s" % (AD, AD2)

    t_abs = np.abs(AD)
    # // Case 1.
    rhs = EA + np.dot(absC, EB)  # np.matmul(absC, EB)
    # if t_abs[0] > rhs[0] or t_abs[1] > rhs[1] or t_abs[2] > rhs[2]:
    #     return False
    # print(rhs[0].dtype)
    # print(t_abs[0].dtype)
    if t_abs[0] > rhs[0]:
        return 0
    if t_abs[1] > rhs[1]:
        return 0
    if t_abs[2] > rhs[2]:
        return 0

    # // Case 2.
    ra = np.dot(EA.T, absC).T
    if np.any(np.abs(np.dot(D.T, B).T) > ra + EB):
        return 0

    # // Case 3.
    dra = EA[1] * absC[2, 0] + EA[2] * absC[1, 0]
    drb = EB[1] * absC[0, 2] + EB[2] * absC[0, 1]
    drt = AD[2] * C[1, 0] - AD[1] * C[2, 0]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 6.
    dra = EA[0]*absC[2, 0] + EA[2]*absC[0, 0]
    drb = EB[1]*absC[1, 2] + EB[2]*absC[1, 1]
    drt = AD[0] * C[2, 0] - AD[2]*C[0, 0]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 11.
    dra = EA[0]*absC[1, 2] + EA[1]*absC[0, 2]
    drb = EB[0]*absC[2, 1] + EB[1]*absC[2, 0]
    drt = AD[1] * C[0, 2] - AD[0]*C[1, 2]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 4.
    dra = EA[1]*absC[2, 1] + EA[2]*absC[1, 1]
    drb = EB[0]*absC[0, 2] + EB[2]*absC[0, 0]
    drt = AD[2] * C[1, 1] - AD[1]*C[2, 1]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 5.
    dra = EA[1]*absC[2, 2] + EA[2]*absC[1, 2]
    drb = EB[0]*absC[0, 1] + EB[1]*absC[0, 0]
    drt = AD[2] * C[1, 2] - AD[1]*C[2, 2]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 7.
    dra = EA[0] * absC[2, 1] + EA[2]*absC[0, 1]
    drb = EB[0]*absC[1, 2] + EB[2]*absC[1, 0]
    drt = AD[0] * C[2, 1] - AD[2]*C[0, 1]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 8.
    dra = EA[0]*absC[2, 2] + EA[2]*absC[0, 2]
    drb = EB[0]*absC[1, 1] + EB[1]*absC[1, 0]
    drt = AD[0] * C[2, 2] - AD[2]*C[0, 2]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 9.
    dra = EA[0]*absC[1, 0] + EA[1]*absC[0, 0]
    drb = EB[1]*absC[2, 2] + EB[2]*absC[2, 1]
    drt = AD[1] * C[0, 0] - AD[0]*C[1, 0]
    if np.abs(drt) > dra + drb:
        return 0

    # // Case 10.
    dra = EA[0]*absC[1, 1] + EA[1]*absC[0, 1]
    drb = EB[0]*absC[2, 2] + EB[2]*absC[2, 0]
    drt = AD[1] * C[0, 1] - AD[0]*C[1, 1]
    if np.abs(drt) > dra + drb:
        return 0

    return 1


class Obb(object):
    __corners = [(-1, -1, -1), (-1, -1, 1),
                 (-1, 1, 1), (-1, 1, -1),
                 (1, 1, -1), (1, -1, -1),
                 (1, -1, 1), (1, 1, 1)]
    __face_ids = [(0, 3, 1), (3, 2, 1),
                  (0, 1, 5), (1, 6, 5),
                  (4, 5, 6), (4, 6, 7),
                  (3, 4, 2), (4, 7, 2),
                  (4, 3, 5), (3, 0, 5),
                  (6, 2, 7), (6, 1, 2)]

    def __init__(self, centroid=None, axes=None, scales=None):
        self._centroid = np.asarray(centroid, dtype=np.float32).reshape((3, 1)) \
            if centroid is not None \
            else np.zeros(shape=(3, 1), dtype=np.float32)
        self._axes = axes if axes is not None \
            else np.identity(3, dtype=np.float32)  # axes in cols
        self._scales = np.asarray(scales, dtype=np.float32).reshape((3, 1)) \
            if scales is not None \
            else np.ones(shape=(3, 1), dtype=np.float32)
        """Full side length"""
        self._corners_3d = None
        self._faces_3d = None

    @classmethod
    def corners(cls):
        return cls.__corners

    @classmethod
    def face_ids(cls):
        return cls.__face_ids

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, vec):
        self._centroid = np.asarray(vec, dtype=np.float32).reshape((3, 1))
        self._corners_3d = None

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        assert axes.shape == (3, 3), \
            "Not allowed shape: %s" % axes.shape.__repr__()
        self._axes = axes
        self._corners_3d = None

    def axes_scaled(self):
        return np.matmul(np.diag(self.scales.flatten()), self.axes)

    @property
    def scales(self):
        """Full side lengths, not half-axis lengths."""
        return self._scales

    @scales.setter
    def scales(self, scales):
        self._scales = np.asarray(scales).reshape((3, 1))
        self._corners_3d = None

    def axis(self, axis_id):
        assert self._scales.shape == (3, 1), \
            "[SceneObj::axis] wrong _scales shape: %s" % \
            self._scales.shape.__repr__()
        assert self._axes.shape == (3, 3), \
            "[SceneObj::axis] wrong _axes shape: %s" % \
            self._axes.shape.__repr__()
        return self._scales[axis_id] * np.squeeze(self._axes[:, axis_id])

    def set_axis(self, axis_id, value):
        length = np.linalg.norm(value)
        self._axes[:, axis_id] = (value / length).astype(np.float32)
        self._scales[axis_id] = np.float32(length)
        assert self._axes.shape == (3, 3), \
            "[sceneObj::set_axis] shape changed: %s" % \
            self._axes.shape.__repr__()

    def corners_3d(self):
        """
        :return: Corners in rows x 3
        """
        if self._corners_3d is not None:
            return self._corners_3d
        else:
            half_axes = np.zeros(shape=(3, 3), dtype="float")
            for a in range(3):
                half_axes[:, a] = self.axis(axis_id=a) / 2.
            corners_3d = \
                np.zeros((len(Obb.corners()), 3), np.float32)
            for row, corner in enumerate(Obb.corners()):
                corners_3d[row, :] = \
                    self.centroid.T \
                    + half_axes[:, 0] * corner[0] \
                    + half_axes[:, 1] * corner[1] \
                    + half_axes[:, 2] * corner[2]
            # print("corners_3d: %s" % corners_3d)
            self._corners_3d = corners_3d
            return corners_3d

    def corners_3d_lower(self, up_axis=(0., -1., 0.)):
        """Returns the 4 points that have smaller y coordinates given up_axis"""
        c3d = self.corners_3d()
        dots = np.dot(c3d, up_axis)
        indices = np.argsort(dots)
        # return c3d[sorted(indices[:4]), :]
        corners_tmp = c3d[sorted(indices[:4]), :]
        for i in range(0, 4):
            i1 = i + 1
            if i1 > 3:
                i1 -= 4
            i2 = i + 2
            if i2 > 3:
                i2 -= 4
            len_diag = np.linalg.norm(corners_tmp[i2, :] - corners_tmp[i, :])
            len_side = np.linalg.norm(corners_tmp[i1, :] - corners_tmp[i, :])
            if len_side > len_diag:
                corners_tmp[i1, :], corners_tmp[i2, :] = \
                    corners_tmp[i2, :], corners_tmp[i1, :].copy()
                # assert corners_tmp[i1] != corners_tmp[i2], \
                #     "Wrong: %s" % corners_tmp
        return corners_tmp

    def faces_3d(self):
        corners_3d = self.corners_3d()
        assert corners_3d.shape[1] == 3, \
            "assumed Nx3: %s" % corners_3d.shape
        faces_3d = np.zeros(shape=(3, 3, len(Obb.face_ids())))
        for face_id, face in enumerate(Obb.face_ids()):
            for d in range(3):
                faces_3d[d, :, face_id] = \
                    corners_3d[face[d], :]
        return faces_3d

    def faces_3d_memoized(self):
        if self._faces_3d is None:
            corners_3d = self.corners_3d()
            self._faces_3d = np.zeros(shape=(3, 3, len(Obb.face_ids())))
            for face_id, face in enumerate(Obb.face_ids()):
                for d in range(3):
                    self._faces_3d[d, :, face_id] = \
                        corners_3d[face[d], :]
        return self._faces_3d

    def rectangles_3d(self):
        corners_3d = self.corners_3d()
        assert corners_3d.shape[1] == 3, \
            "assumed Nx3: %s" % corners_3d.shape
        return np.array(
          [corners_3d[[3, 2, 1, 0], :],
           corners_3d[4:8, :],
           corners_3d[[0, 5, 4, 3], :],
           corners_3d[[6, 1, 2, 7], :],
           corners_3d[[3, 4, 7, 2], :],
           corners_3d[[0, 1, 6, 5], :]])

    def to_obj_string(self, name, vertex_offset=0):
        lines = []
        lines.append("o %s\n" % name)
        corners_3d = self.corners_3d()
        for row in range(corners_3d.shape[0]):
            lines.append("v %f %f %f\n" %
                         (corners_3d[row, 0],
                          corners_3d[row, 1],
                          corners_3d[row, 2]))

        # lines.append("usemtl None\ns off\n")
        # print("vertex_offset: %d" % vertex_offset)
        for face in Obb.face_ids():
            lines.append("f %d %d %d\n" %
                         (vertex_offset+face[0]+1, vertex_offset+face[2]+1,
                          vertex_offset+face[1]+1))
        return "".join(lines)

    def save(self, path, name, part_id=None, save_obj=False):
        """ Saves a cuboid to an obj file.
        :param path: Output path ending in ".obj" 
        :param name: Name of object
        """
        # OBB obj
        out_path = None
        target_path = path + ('.obj' if path[-3:] != 'obj' else '')
        if save_obj:
            with open(target_path, 'w') as fout:
                fout.write(self.to_obj_string(name))
        out_path = target_path
        with codecs.open(path + '.json', 'w', encoding='utf-8') as fout:
            j_out = self.to_json(part_id=part_id)
            json.dump(j_out, fout,
                      separators=(',', ':'), sort_keys=True, indent=4)
        return out_path

    def to_json(self, part_id=None):
        d = {'scales':  self.scales.tolist(),
             'centroid': self.centroid.tolist(),
             'axes': self.axes.tolist()}
        if part_id is not None:
            d['part_id'] = part_id
        return d

    @classmethod
    def from_json(cls, data):
        obb = Obb()
        obb.scales = np.asarray(data['scales'])
        obb.centroid = np.asarray(data['centroid'])
        obb.axes = np.asarray(data['axes'])
        return obb

    @classmethod
    def load(cls, path, return_data=False):
        with codecs.open(path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
            if return_data:
                return cls.from_json(data), data
            else:
                return cls.from_json(data)

    def apply_transform(self, transform):
        """Applies transform to self"""
        tmp_shape = self.centroid.shape
        # self.centroid = \
        #     geometry.hnormalized(
        #         np.dot(transform, geometry.homogeneous(self.centroid)))
        self.centroid = geometry.htransform(transform, self._centroid)
        assert tmp_shape == self.centroid.shape, \
            "Changed shape after transform: %s => %s" % \
            (tmp_shape, self.centroid.shape)
        self.axes = np.dot(transform[:3, :3], self._axes)
        # geometry.hnormalized(
        #    np.dot(transform,
        #           geometry.homogeneous(self.axes, is_vector=True)))
        assert self.axes.shape == (3, 3), \
            "Axes shape changed: %s" % self.axes.__repr__()
        self._corners_3d = None

    def as_transform(self):
        """
        :returns: A 4x4 homogeneous transform.
        """
        a = self._axes
        c = self._centroid
        return np.array(
            ((a[0][0], a[0][1], a[0][2], c[0]),
             (a[1][0], a[1][1], a[1][2], c[1]),
             (a[2][0], a[2][1], a[2][2], c[2]),
             (0., 0., 0., 1.)),
            dtype=np.float32)
        # return np.hstack((np.vstack((self.axes,
        #                              [0., 0., 0.])),
        #                   np.vstack((self.centroid,
        #                              1.))
        #                   ))

    # def __str__(self):
    #     return "Obb(%s, %s, %s)" % (self.centroid.T, self.axes, self.scales.T)

    def __eq__(self, other):
        return np.allclose(self.centroid, other.centroid) \
               and np.allclose(self.axes, other.axes) \
               and np.allclose(self.scales, other.scales)

    def point_close_to(self, point, dist_thresh):
        """
        Checks, if a 3D point is close to any of the faces.
        :param point: 3D query point.
        :param dist_thresh: A face is close, if the point-to-face distance is 
                            less, than dist_thresh.
        :return bool: True, if close 
        """
        obb_faces_3d = self.faces_3d()
        for face_id in range(obb_faces_3d.shape[2]):
            face = np.asarray(obb_faces_3d[:, :, face_id].squeeze())
            dist, _ = pointTriangleDistance(face, np.asarray(point))
            if dist < dist_thresh:
                return True
        return False

    def closest_face_dist_memoized(self, point):
        """
        Checks, if a 3D point is close to any of the faces.
        :param point: 3D query point.
        :param dist_thresh: A face is close, if the point-to-face distance is
                            less, than dist_thresh.
        :return bool: True, if close
        """
        mn_dist = 1.e9
        obb_faces_3d = self.faces_3d_memoized()
        for face_id in range(obb_faces_3d.shape[2]):
            face = np.array(obb_faces_3d[:, :, face_id].squeeze())
            dist, _ = pointTriangleDistance(face, np.asarray(point))
            if dist < mn_dist:
                mn_dist = dist
        return mn_dist

    # _isec_stats = np.zeros(11, dtype=int)

    @jit(cache=True)
    def intersects2(self, other):
        return _intersects(self.centroid, self._axes, self.scales,
                           other.centroid, other._axes, other.scales)

    @jit(cache=True)
    def intersects(self, other):
        """
        Source: https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf
        :param other: Other obb to intersect
        :return: 
        """
        # A = gm.normalized(self.axes, axis=0)
        A = self._axes
        EA = self.scales / 2.

        # B = gm.normalized(other.axes, axis=0)
        B = other._axes
        EB = other.scales / 2.

        C = np.matmul(A.T, B)  # c_ij in pdf
        AbsC = np.abs(C)
        # assert np.all(AbsC >= 0.), "Not abs: %s" % AbsC

        # // Compute the translation vector.
        D = other.centroid - self.centroid  # D in pdf
        # assert D.shape == (3, 1), "D wrong shape: %s" % D.shape.__repr__()

        AD = np.matmul(A.T, D)
        # assert AD.shape == (3, 1), "t wrong shape: %s" % AD.shape.__repr__()

        t_abs = np.abs(AD)
        # // Case 1.
        rhs = EA + np.matmul(AbsC, EB)
        if t_abs[0] > rhs[0] or t_abs[1] > rhs[1] or t_abs[2] > rhs[2]:
            # print("Case 1")
            # Obb._isec_stats[0] += 1
            return False

        # // Case 2.
        ra = np.matmul(EA.T, AbsC).T
        if np.any(np.abs(np.matmul(D.T, B).T) > ra + EB):
            # print("Case 2")
            # Obb._isec_stats[1] += 1
            return False

        # // Case 3.
        # double dra = ext(1)*AbsR(2, 0) + ext(2)*AbsR(1, 0);
        dra = EA[1] * AbsC[2, 0] + EA[2] * AbsC[1, 0]
        # double drb = EB(1)*AbsR(0, 2) + EB(2)*AbsR(0, 1);
        drb = EB[1] * AbsC[0, 2] + EB[2] * AbsC[0, 1]
        # double drt = t(2) * C(1, 0) - t(1)*C(2, 0);
        drt = AD[2] * C[1, 0] - AD[1] * C[2, 0]
        if abs(drt) > dra + drb:
            # print("Case 3")
            # Obb._isec_stats[2] += 1
            return False

        # // Case 6.
        dra = EA[0]*AbsC[2, 0] + EA[2]*AbsC[0, 0]
        drb = EB[1]*AbsC[1, 2] + EB[2]*AbsC[1, 1]
        drt = AD[0] * C[2, 0] - AD[2]*C[0, 0]
        if abs(drt) > dra + drb:
            # print("Case 6.")
            # Obb._isec_stats[3] += 1
            return False

        # // Case 11.
        dra = EA[0]*AbsC[1, 2] + EA[1]*AbsC[0, 2]
        drb = EB[0]*AbsC[2, 1] + EB[1]*AbsC[2, 0]
        drt = AD[1] * C[0, 2] - AD[0]*C[1, 2]
        if abs(drt) > dra + drb:
            # print("Case 11.")
            # Obb._isec_stats[4] += 1
            return False

        # // Case 4.
        dra = EA[1]*AbsC[2, 1] + EA[2]*AbsC[1, 1]
        drb = EB[0]*AbsC[0, 2] + EB[2]*AbsC[0, 0]
        drt = AD[2] * C[1, 1] - AD[1]*C[2, 1]
        if abs(drt) > dra + drb:
            # print("Case 4")
            # Obb._isec_stats[5] += 1
            return False

        # // Case 5.
        dra = EA[1]*AbsC[2, 2] + EA[2]*AbsC[1, 2]
        drb = EB[0]*AbsC[0, 1] + EB[1]*AbsC[0, 0]
        drt = AD[2] * C[1, 2] - AD[1]*C[2, 2]
        if abs(drt) > dra + drb:
            # print("Case 5.")
            # Obb._isec_stats[6] += 1
            return False

        # // Case 7.
        dra = EA[0] * AbsC[2, 1] + EA[2]*AbsC[0, 1]
        drb = EB[0]*AbsC[1, 2] + EB[2]*AbsC[1, 0]
        drt = AD[0] * C[2, 1] - AD[2]*C[0, 1]
        if abs(drt) > dra + drb:
            # print("Case 7.")
            # Obb._isec_stats[7] += 1
            return False

        # // Case 8.
        dra = EA[0]*AbsC[2, 2] + EA[2]*AbsC[0, 2]
        drb = EB[0]*AbsC[1, 1] + EB[1]*AbsC[1, 0]
        drt = AD[0] * C[2, 2] - AD[2]*C[0, 2]
        if abs(drt) > dra + drb:
            # print("Case 8.")
            # Obb._isec_stats[8] += 1
            return False

        # // Case 9.
        dra = EA[0]*AbsC[1, 0] + EA[1]*AbsC[0, 0]
        drb = EB[1]*AbsC[2, 2] + EB[2]*AbsC[2, 1]
        drt = AD[1] * C[0, 0] - AD[0]*C[1, 0]
        if abs(drt) > dra + drb:
            # print("Case 9.")
            # Obb._isec_stats[9] += 1
            return False

        # // Case 10.
        dra = EA[0]*AbsC[1, 1] + EA[1]*AbsC[0, 1]
        drb = EB[0]*AbsC[2, 2] + EB[2]*AbsC[2, 0]
        drt = AD[1] * C[0, 1] - AD[0]*C[1, 1]
        if abs(drt) > dra + drb:
            # print("Case 10.")
            # Obb._isec_stats[10] += 1
            return False

        return True

    _unit_vecs = np.identity(3)

    @classmethod
    def from_vector(cls, vec, scale=0.1):
        obb = Obb(centroid=vec/2.)
        # obb.centroid = vec / 2.
        obb.set_axis(0, vec)
        unit_vec = obb._axes[:, 0]
        best_unit_axis_col = min(enumerate(
            [np.abs(np.dot(Obb._unit_vecs[:, col], unit_vec))
             for col in range(3)]), key=lambda e: e[1])[0]
        obb.set_axis(1, np.cross(unit_vec, Obb._unit_vecs[:, best_unit_axis_col]))
        obb.set_axis(2, np.cross(unit_vec, obb.axis(1)))
        obb.scales[2] = obb.scales[1] = scale
        # obb.scales[2] = scale
        return obb

    def volume(self):
        # corners_3d = self.corners_3d()
        # diag = corners_3d[-1, :] - corners_3d[0, :]
        # return diag[0] * diag[1] * diag[2]
        scales = self.scales
        return scales[0] * scales[1] * scales[2]


class SceneObjPart(object):
    __colors = None

    def __init__(self, label):
        if not self.__colors:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(curr_path, '..', 'colors.json')) as f:
                self.__colors = json.load(f)

        self.label = label
        self.vertices = {}
        self.faces = []  # refers to original vids
        self.obb = None
        self.path = None

    # @property
    # def label(self):
    #     return self._label
    #
    # @label.setter
    # def label(self, label):
    #     self._label = label

    # @property
    # def vertices(self):
    #     return self._vertices

    # @property
    # def faces(self):
    #     return self._faces

    # @property
    # def obb(self):
    #     return self._obb
    #
    # @obb.setter
    # def obb(self, obb):
    #     self._obb = obb
    #     assert self.obb is not None, \
    #         "Could not set obb: %s from %s" % (self.obb, obb)

    def add_vertex(self, vid, pnt):
        if vid in self.vertices:
            raise RuntimeError("Vertex id %d already stored" % vid)
        self.vertices[vid] = pnt

        return True

    def get_vertex_pos(self, vid):
        return self.vertices[vid][:3]

    def add_face(self, face):
        assert len(face) == 3, \
            "Need three ints: %s" % face.__repr__()
        self.faces.append(face)

    def get_face_3d(self, face_id):
        assert face_id < len(self.faces), \
            "face_id %d >= %d len(faces)" % \
            (face_id, len(self.faces))
        face = self.faces[face_id]
        out = []
        for vid in face:
            out.append(self.get_vertex_pos(vid))
        return out

    def save(self, path, part_id=None, save_obj=False):
        out_paths = {'obbs': [], 'clouds': []}
        # OBB Obj
        if self.obb:
            out_paths['obbs'].append(
               self.obb.save(path, self.label, part_id=part_id,
                             save_obj=save_obj))
        else:
            warnings.warn("No obb for part %s" % self.label)

        # Mesh PLY
        if len(self.vertices):
            cloud_path = path + '.ply'
            with open(cloud_path, 'w') as fout:
                fout.write("ply\n"
                           "format ascii 1.0\n"
                           "element vertex %d\n"
                           "property float x\n"
                           "property float y\n"
                           "property float z\n"
                           "property float nx\n"
                           "property float ny\n"
                           "property float nz\n"
                           "property uchar red\n"
                           "property uchar green\n"
                           "property uchar blue\n"
                           "property uchar alpha\n"
                           "element face %d\n"
                           "property list uchar int vertex_indices\n"
                           "end_header\n" % (
                               len(self.vertices),
                               len(self.faces)))
                vids_new = {}
                vid_lin = 0
                for vid in self.vertices:
                    v = self.vertices[vid]
                    fout.write("%f %f %f %f %f %f %d %d %d %d\n" % (
                        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]
                    ))
                    vids_new[vid] = vid_lin
                    vid_lin += 1

                for face in self.faces:
                    face_lin = [vids_new[face[0]],
                                vids_new[face[1]],
                                vids_new[face[2]]]
                    for d in range(3):
                        if face_lin[d] >= vid_lin:
                            warnings.warn("face_lin[0]: %d > %d n_points"
                                          % (face_lin[d], vid_lin))
                    try:
                        fout.write("3 %d %d %d\n" % (
                            face_lin[0], face_lin[1], face_lin[2]
                        ))
                    except KeyError:
                        print("Could not look up face %s"
                              % (face))
                        print("%s" % [vid for vid in vids_new])

                out_paths['clouds'].append(cloud_path)
                # print("out_paths is now: %s" % out_paths)
        return out_paths

    def apply_transform(self, transform):
        self.obb.apply_transform(transform)
        for _, vertex in self.vertices.items():
            v = np.array(vertex[:3])
            # assert v_hom.shape[0] == 4 and v_hom.shape[1] == 1, \
            #     "shape not 4x1: %s" % v_hom.shape.__repr__()
            vertex[:3] = (np.dot(transform[:3, :3], v.T) + transform[:3, 3]).T
            vertex[3:6] = np.dot(transform[:3, :3],
                                 np.array(vertex[3:6]).reshape((3, 1)))


class SceneObj(SceneObjectInterface):
    def __init__(self, label):
        try:
            super(SceneObj, self).__init__(label)
        except TypeError:
            self.label = label
        self._parts = {}
        # self._obb = None

    # @property
    # def label(self):
    #     return self._label
    #
    # @label.setter
    # def label(self, label):
    #     self._label = label

    @property
    def parts(self):
        return self._parts

    def add_part(self, part_id, label_or_part):
        """Adds a SceneObjPart to the SceneObj instance.

        Arguments:
            part_id (int):
                Unique identifier of part inside this object.
                Will assign the next one automatically, if negative.
            label_or_part (str, SceneObjPart):
                Category of part, or the part to add.
        Returns:
            Reference to the part added.
        """
        # assign part_id automatically
        if part_id < 0:
            part_id = len(self.parts)
        # ensure uniqueness of id
        if part_id in self._parts:
            raise RuntimeError("Part %d already added (%s)"
                               % (part_id, self.get_part(part_id).get_label()))

        # if label_or_part is a label (category)
        if isinstance(label_or_part, str) \
                or type(label_or_part).__name__ == 'unicode':
            # store category
            label = label_or_part
            # find the next unique label for this part by appending an int
            # if such a part already exists (e.g. armrest-1)
            lin_id = 0
            while True:
                # same names
                if next((p for p in self.parts.values() if p.label == label),
                        None) is None:
                    break
                else:
                    lin_id += 1
                    label = "%s-%d" % (label_or_part, lin_id)
            self._parts[part_id] = SceneObjPart(label)
        elif isinstance(label_or_part, SceneObjPart) \
            or type(label_or_part) == SceneObjPart \
            or 'SceneObjPart' in '{}'.format(type(label_or_part)):
            self._parts[part_id] = label_or_part
        else:
            print('isinstance: {}'
                  .format(isinstance(label_or_part, SceneObjPart)))
            print('str: ', '{}'.format(type(label_or_part)))
            raise TypeError("Unrecognized type: %s" % type(label_or_part))

        return self._parts[part_id]

    def has_part(self, part_id):
        return part_id in self._parts

    def get_part(self, part_id):
        return self._parts[part_id]

    def get_part_by_name(self, substr):
        return next(
            (part for part in self.parts.values()
             if substr in part.label),
            None)
    def get_part_by_name_strict(self, substr):
        return next(
          (part for part in self.parts.values()
           if substr == part.label),
          None)

    def __getitem__(self, index):
        # http://stackoverflow.com/questions/19151/build-a-basic-python-iterator
        return list(self._parts)[index]

    def add_vertex(self, part_id, vid, pnt):
        if part_id not in self._parts:
            raise RuntimeError(
                "Please add part %d before adding vertices to it" % part_id)
        return self._parts[part_id].add_vertex(vid, pnt)

    def save(self, path_prefix, save_obj=False):
        """Save SceneObject to disk.

        Args:
            path_prefix (str):
                Destination path and name of the object prepended to the
                name of parts.
            save_obj (bool):
                Save obj-s not just json for objects.
        """
        out_paths = {}
        for part_id, part in self.parts.items():
            added_paths = \
                part.save(path_prefix + "_%s" % part.label, part_id=part_id,
                          save_obj=save_obj)
            # print("[SceneObj::save()] added paths: %s" % added_paths)
            for k, v in added_paths.items():
                if k in out_paths:
                    out_paths[k].extend(v)
                else:
                    out_paths[k] = v
        return out_paths

    def get_name(self, obj_id):
        return "%02d_%s" % (obj_id, self.label)

    # @override
    def apply_transform(self, transform):
        """ Transforms all parts and OBB-s with transform
        :param transform: Homogeneous transformation matrix, 
                          will be pre-multiplied.
        :return: None
        """
        for part_id, part in self.parts.items():
            part.apply_transform(transform)

    def __repr__(self):
        return "SceneObj(%s with %d parts)" % (self.label, len(self.parts))

    def point_close_to_obb(self, point, dist_thresh):
        """
        Checks, if a 3D point is close to any of the parts.
        :param point: 3D query point.
        :param dist_thresh: A face is close, if the point-to-face distance is 
                            less, than dist_thresh.
        :return bool: True, if close 
        """
        for part in self.parts.values():
            if part.obb.point_close_to(point, dist_thresh):
                return True
        return False

    # @override
    def get_centroid(self, squeeze=False):
        c = np.sum(part.obb.centroid for part in self._parts.values()) \
            / len(self._parts)
        return np.squeeze(c) if squeeze else c

    # @override
    def closer_to_scene_object_than(self, other, max_dist):
        centroid = self.centroid()
        for part_other in other._parts.values():
            corners_other = part_other.obb.corners_3d()
            for corner in range(corners_other.shape[0]):
                diff = corner - centroid
                if np.sqrt(diff[0]*diff[0]
                           + diff[1]*diff[1]
                           + diff[2]*diff[2]) < max_dist:
                    return True
        return False

    # @override
    def get_transform(self):
        """Returns local->world transform of this object"""
        return self._largest_part().obb.as_transform()

    def get_angle(self, positive_only):
        """
        Gets angle of Pigraphs object
        :param positive_only: Offsets from -pi..pi to 0..2pi
        :return:
        """
        tr = self.get_transform()
        # print("transform: %s" % tr)
        # scenelets are aligned to have "-y" as forward,
        # but for some reason they have +y as forward...
        # return np.arccos(np.dot(unit_vector, tr[:3, 1]))
        if positive_only:
            ang = -np.arctan2(tr[2, 1], tr[0, 1]) + np.pi
            if ang >= TWO_PI:
                return ang - TWO_PI
            else:
                return ang
        else:
            return -np.arctan2(tr[2, 1], tr[0, 1])

    def intersects(self, other):
        for part0 in self.parts.values():
            for part1 in other.parts.values():
                if part0.obb.intersects(part1.obb):
                    return True
        return False

    def _largest_part(self, contains={'seat', 'top'}, with_part_id=False):
        """Returns part with largest volume bounding box"""
        if contains is not None:
            try:
                item = \
                    max(((k, v) for k, v in self._parts.items()
                         if any(part_name in v.label
                                for part_name in contains)),
                        key=lambda part: part[1].obb.volume())
            except ValueError:
                # print("did not find seat for %s" % self.label)
                item = max(self._parts.items(),
                           key=lambda part: part[1].obb.volume())
        else:
            item = max(self._parts.items(),
                       key=lambda part: part[1].obb.volume())
        return item[1] if not with_part_id else item

    def to_obj_string(self, prefix="", vertex_offset=0):
        return "".join(
            [
                part.obb.to_obj_string(name="_".join((prefix, part.label)),
                                       vertex_offset=vertex_offset + i*8)
                for i, part in enumerate(self._parts.values())
            ]
        )

    @lru_cache(maxsize=1)
    def get_hull_of_part_obbs(self):
        """Returns the convex hull of part obbs"""
        from imapper.logic.mesh_OBJ import MeshOBJ
        verts = [part.obb.corners_3d() for part in self.parts.values()]
        return MeshOBJ.hull_from_vertices(np.concatenate(verts, axis=0))

    @lru_cache(maxsize=1)
    def get_centroid_from_hull(self):
        return self.get_hull_of_part_obbs().get_centroid()

