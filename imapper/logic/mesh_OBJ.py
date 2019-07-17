import copy  # deepcopy
import json
import subprocess  # save Scene
import sys
import warnings

import numpy as np
import os
# from numba import jit

# from visualizer.visualizer import Visualizer
from imapper.logic.scene_object import Obb
from imapper.logic.scene_interface import SceneInterface
from imapper.logic.scene_object_interface import SceneObjectInterface

if not sys.version_info[0] < 3:
    from functools import lru_cache
else:
    from repoze.lru import lru_cache


class Scene(SceneInterface):
    def __init__(self, name):
        super(Scene, self).__init__(name)

    def add_object(self, object_id, obj, clone):
        if object_id < 0:
            object_id = len(self.objects)
        if object_id in self.objects:
            sys.stderr.write("Already have object with id %d\n" % object_id)
            raise RuntimeError("NOOO %d" % object_id)
        if clone:
            self._objects[object_id] = copy.deepcopy(obj)
        else:
            self._objects[object_id] = obj

        return self.objects[object_id]

    # @override
    def save(self, path):
        #path.split(os.sep)
        name_scene = \
            os.path.split(os.path.dirname(os.path.abspath(os.path.join(path, os.pardir))))[1].split('_')[0]
        tmp_dir = os.path.join(os.path.dirname(path), "%s_objects" % name_scene)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            pass
        assert os.path.isdir(tmp_dir), "not dir: %s" % tmp_dir
        names = []
        # oris = dict()
        transforms = dict()
        for oid, obj in enumerate(self.objects.values()):
            name_obj = "%02d_%s_%s.obj" % (oid, obj.label, obj.name)
            names.append(os.path.join(tmp_dir, name_obj))
            obj.save(names[-1])
            # oris[name_obj] = obj.get_angle()
            transforms[name_obj] = obj.transform.tolist()
            obj.get_obb()
        # cmd = ("meshlabserver -i %s -o %s "
        #        "-s /home/amonszpa/workspace/stealth/scripts/meshlab/flatten_layers.mlx "
        #        "-om vc vn >/dev/null" % (" ".join(names), path))
        # print("calling %s" % cmd)
        # subprocess.call(cmd.split(" "))
        # for name in names:
        #     os.remove(name)
        # os.rmdir(tmp_dir)

        scenelet = {'3d': {}, 'preproc': {}, 'scenelets': {'obbs': []}}
        # scenelet['scenelets']['obbs'].append(os.path.split(path)[1])
        scenelet['scenelets']['obbs'] = [os.path.join("%s_objects" % name_scene, 
                                                      os.path.split(name)[1]) 
                                         for name in names]
        # scenelet['scenelets']['orientations'] = oris
        scenelet['scenelets']['transforms'] = transforms
        json.dump(scenelet,
                  fp=open("%s.json" % os.path.splitext(path)[0], 'w'),
                  indent=2)
        # assert os.path.exists(path), "Did not work: %s" % path


class SceneObject(SceneObjectInterface):
    """Abstract object class wrapping a meshOBJ.
    Note: pigraphs.SceneObj and action_synth.SceneObjAS do the same, 
    but are special, and this is hopefully more general. 
    """
    def __init__(self, label, name=None, mesh=None):
        """Constructor"""
        # TODO: fix
        print("[mesh_OBJ.py::SceneObject::__init__] Hack here")
        # super(logic.mesh_OBJ.SceneObject, self).__init__(label)
        # super(SceneObject, self).__init__(label)
        self.label = label

        self.name = name
        """Hopefully unique identifier for object"""
        self._mesh = mesh
        """MeshOBJ internal storage"""

        self.transform = np.identity(4, dtype=np.float32)
        """The "pose" of the mesh recording the transformations applied to it,
        since it was created."""

    # @override
    def apply_transform(self, transform, update=True):
        """
        :param transform:
        :param update: Should the internal relative transform be updated?
        :return:
        """
        # try:
        self._mesh.apply_transform(transform)
        if update:
            self.transform = np.matmul(transform, self.transform)
        # except AttributeError:
        #     sys.stderr.write("No mesh to apply to...\n")

    # @override
    def get_centroid(self):
        # try:
        return self._mesh.get_centroid()
        # except AttributeError:
        #     sys.stderr.write("No mesh to apply to...\n")

    # @override
    def closer_to_scene_object_than(self, other, max_dist):
        raise NotImplementedError("todo...")

    # @override
    def get_transform(self):
        try:
            return self._mesh.get_obb().get_transform()
        except AttributeError:
            sys.stderr.write("No mesh to apply to...\n")

    def get_angle(self, positive_only):
        tr = self.transform
        # ActionSynth objects are designed to have negative y forward
        # They are also rotated upon read in in main_sampling.py, so that
        # their new forward is -z
        # return np.arccos(np.dot(-tr[:3, 2], unit_vector))
        if positive_only:
            return np.float32(-np.arctan2(tr[2, 2], tr[0, 2]) + np.pi)
        else:
            return np.float32(-np.arctan2(tr[2, 2], tr[0, 2]))

    def save(self, path):
        try:
            if not path.endswith('.obj'):
                path = "%s.obj" % path
            self._mesh.save_obj(path)
        except AttributeError:
            sys.stderr.write("No mesh to apply to...\n")
        return path

    # @jit(cache=True)
    def intersects(self, other):
        return self._mesh.intersects(other._mesh)

    # @jit(cache=True)
    def get_angles_euler(self, which=-1):
        """Return euler angles"""
        T = self.transform
        if which == 1:  # angle around Y
            return np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
        elif which == -1:  # Euler-XYZ
            return (np.arctan2(T[2, 1], T[2, 2]),
                    np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2)),
                    np.arctan2(T[1, 0], T[0, 0]))
        elif which == 0:  # angle around X
            return np.arctan2(T[2, 1], T[2, 2])
        elif which == 2:  # angle around Z
            return np.arctan2(T[1, 0], T[0, 0])

    def get_name(self, obj_id):
        """Used by Scenelet.save"""
        return "%02d_%s" % (obj_id, self.name)

    def get_obb(self):
        return self._mesh.get_obb()

    @property
    def mesh(self):
        return self._mesh


class _Shape(object):
    """Internal class of MeshOBJ to represent a set of faces,
    corresponding to a group in a Wavefront OBJ file
    """
    def __init__(self, name):
        self.name = name
        """Name of the group"""
        self.faces = None
        """Mx3 vertex index matrix (ints)"""


class MeshOBJ(object):
    """Mesh class that is arranged to be easily convertible to Wavefront OBJ"""
    def __init__(self, path=None, verbose=True):
        self._hull = None
        """3D convex _hull mesh"""
        self._obb = None
        """Oriented bounding box of mesh"""

        self.path = None
        """Path that we loaded from"""

        if path is None:
            self.vertices = None
            """Nx3 vertex matrix"""
            self.normals = None
            """Nx3 normal matrix"""
            self.shapes = dict()
            """Dict of sub-objects (obj wavefront groups)"""
        else:
            assert os.path.exists(path), "No: %s" % path
            if verbose:
                print("[MeshOBJ] Loading %s" % path)
            self.vertices, self.normals, self.shapes = \
                MeshOBJ._load_from_disk(path)
            self.path = path

    @classmethod
    @lru_cache(maxsize=256)
    def _load_from_disk(cls, path):
        import tinyobjloader as tol
        tol_model = tol.LoadObj(path)
        n_verts = int(len(tol_model['attribs']['vertices']) / 3)
        vertices = \
            np.asarray(tol_model['attribs']['vertices'],
                       dtype=np.float32).reshape(n_verts, 3)
        normals = None
        try:
            if len(tol_model['attribs']['normals']):
                normals = \
                    np.asarray(tol_model['attribs']['normals'],
                               dtype=np.float32).reshape(n_verts, 3)
        except ValueError:
            sys.stderr.write("Could not parse mesh normals for %s\n" % path)
            normals = None

        shapes = {}
        for name_shape in tol_model['shapes']:
            tol_shape = tol_model['shapes'][name_shape]
            assert len([n for n in tol_shape['num_face_vertices'] if n != 3]) == 0, \
                "Non triangles detected...need to change the next line"
            shape = _Shape(name_shape)
            # step = len(tol_shape['indices']) / n_verts
            shape.faces = np.asarray(tol_shape['indices'][::3],
                                     dtype=int, order='F') \
                .reshape((-1, 3))
            shapes[name_shape] = shape
        return vertices, normals, shapes

    def get_n_shapes(self):
        return len(self.shapes.keys()) \
            if self.shapes is not None \
            else 0

    # @jit(cache=True)
    def get_centroid(self):
        # NOTE: changed on 28/4/2017
        # return np.mean(self.get_hull().vertices, axis=0)
        return self.get_obb().centroid
        # if self.get_n_shapes() == 1:
        #     mean = np.mean(self.vertices, axis=0)
        #     assert mean.shape == (3,), \
        #         "Wrong shape: %s" % \
        #         mean.shape.__repr__()
        #     return mean
        # else:
        #     raise NotImplementedError("Need to select a shape")

    # @jit(cache=True)
    def apply_transform(self, transform):
        self.vertices = \
            np.matmul(transform[:3, :3], self.vertices.T).T \
            + transform[:3, 3]
        if self.normals is not None:
            self.normals = np.matmul(transform[:3, :3], self.normals.T).T

        # try:
        if self._hull is not None:
            self._hull.apply_transform(transform)
        # except AttributeError:
        #     pass
        # try:
        if self._obb is not None:
            self._obb.apply_transform(transform)
        # except AttributeError:
        #     pass
        # self._set_changed()

    def save_obj(self, path):
        try:
            # print("creating %s" % os.path.dirname(path))
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        with open(path, 'w') as f:
            vertices = self.vertices
            # print("shape: %s" % vertices.shape.__repr__())
            for row in range(self.vertices.shape[0]):
                f.write("v %f %f %f\n" %
                        (vertices[row, 0], vertices[row, 1], vertices[row, 2]))
            if self.normals is not None and self.normals.shape[0] != 0:
                warnings.warn("Normals saving not implemented yet!")

            for name_shape, shape in self.shapes.items():
                f.write("g %s\n" % name_shape)
                faces = shape.faces
                for row in range(shape.faces.shape[0]):
                    f.write("f %d %d %d\n" %
                            (faces[row, 0] + 1,
                             faces[row, 1] + 1,
                             faces[row, 2] + 1))

            print("Wrote to %s" % os.path.abspath(path))

    def get_hull(self, with_vis=False):
        """Estimates convex _hull"""
        from scipy.spatial import ConvexHull
        if self._hull is None:
            qhull = ConvexHull(self.vertices)
            # print(dir(qhull))
            hull_mesh = MeshOBJ()
            hull_mesh.vertices = self.vertices[qhull.vertices]
            remap = dict((vid, i) for i, vid in enumerate(qhull.vertices))
            # print("remap: %s" % remap)
            faces = []
            for simplex in qhull.simplices:
                # print("simplex: %s" % simplex)
                # print([remap[vid] for vid in simplex])
                faces.append([remap[vid] for vid in simplex])
            hull_shape = _Shape("_hull")
            # print("faces: %s" % faces)
            hull_shape.faces = np.asarray(faces, dtype=int)
            hull_mesh.shapes[hull_shape.name] = hull_shape
            # if with_vis:
            #     vis = Visualizer()
            #     vis.add_coords()
            #     vis.add_mesh(self)
            #     vis.add_mesh(hull_mesh, "_hull")
            #     vis.show()
            self._hull = hull_mesh
        return self._hull

    @staticmethod
    def hull_from_vertices(vertices):
        from scipy.spatial import ConvexHull
        qhull = ConvexHull(vertices)
        hull_mesh = MeshOBJ()
        hull_mesh.vertices = vertices[qhull.vertices]
        remap = dict((vid, i) for i, vid in enumerate(qhull.vertices))
        faces = []
        for simplex in qhull.simplices:
            faces.append([remap[vid] for vid in simplex])
        hull_shape = _Shape("_hull")
        hull_shape.faces = np.asarray(faces, dtype=int)
        hull_mesh.shapes[hull_shape.name] = hull_shape
        # hull is itself
        hull_mesh._hull = hull_mesh
        return hull_mesh

    def get_obb(self):
        if self._obb is None:
            self._obb = Obb()
            hull = self.get_hull()
            mn = np.min(hull.vertices, axis=0)
            mx = np.max(hull.vertices, axis=0)
            diag = mx - mn
            self._obb.centroid = (mn + diag / 2.).astype(np.float32)
            assert self._obb.centroid.shape == (3, 1), "Wrong shape: %s" % repr(self._obb.shape)
            assert diag.shape == (3,), "Wrong shape: %s" % repr(diag.shape)
            self._obb.set_axis(0, np.asarray([diag[0], 0., 0.], dtype=np.float32))
            self._obb.set_axis(1, np.asarray([0., diag[1], 0.], dtype=np.float32))
            self._obb.set_axis(2, np.asarray([0., 0., diag[2]], dtype=np.float32))

        return self._obb

    def _set_changed(self):
        self._hull = None
        self._obb = None

    @classmethod
    def from_obb(cls, obb):
        mesh = MeshOBJ()
        mesh.vertices = obb.corners_3d()
        shape = _Shape("obb")
        shape.faces = np.asarray(obb.face_ids(), dtype=int)
        assert shape.faces.shape == (12, 3), "Wrong shape: %s" % repr(shape.faces.shape)
        mesh.shapes["obb"] = shape
        return mesh

    def intersects(self, other):
        try:
            # TODO: use hull-s, if obb-s intersect
            return self._obb.intersects(other._obb)
        except AttributeError:
            return self.get_obb().intersects(other.get_obb())

    def intersects2(self, other):
        try:
            # TODO: use hull-s, if obb-s intersect
            return self._obb.intersects2(other._obb)
        except AttributeError:
            return self.get_obb().intersects2(other.get_obb())
