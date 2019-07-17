import codecs
import copy
# import simplejson as json
import os
import sys
import time
import warnings
import shutil
from collections import Counter

import numpy as np

from imapper.util.json import json
from imapper.logic.geometry import rot_y, translation
from imapper.logic.mesh_OBJ import MeshOBJ, SceneObject
from imapper.logic.scene_object import Obb, SceneObj, SceneObjPart
from imapper.logic.skeleton import Skeleton
from imapper.util.stealth_logging import lg
from imapper.scenelet_fit.consts import HALF_PI_32
import collections
if not sys.version_info[0] < 3:
    from typing import Tuple, Dict, Iterable

if json.__name__ != 'ujson':
    class MyJsonEncode(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, collections.abc.ValuesView):
                return list(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                print("MyJsonEncode: %s" % type(obj))
            return json.JSONEncoder.default(self, obj)
else:
    MyJsonEncode = None


class Scenelet(object):
    __KEYS_SKELETON = frozenset(
      {'3d', 'scenelets', 'forwards', 'angles',
       'visibility', 'charness', 'confidence', 'density',
       'frame_ids', 'charness_poses', 'name_scene',
       'name_scenelet', 'actors', 'confidence_normalized'})
    def __init__(self, skeleton=None):
        self.skeleton = skeleton if skeleton is not None else Skeleton()
        self.objects = {}
        self.aux_info = {}
        self.transforms = {}
        self.name_scene = None
        self.name_scenelet = None
        self.charness = np.float32(1.)
        """Per-scenelet charness (a single float32)."""
        self.match_score = np.float32(0.)
        """Also called feature distance, smaller is better."""
        # self.confidence = None
        # """Array to show how sure we are about the different
        # entries in the skeleton."""
        self.density = None
        """How densely sampled in space this part of the scene is 
        (how many poses there are usually around the middle frame).
        """
        self.charness_poses = None
        """A dictionary that stores a characteristicness for each pose.
        """

    @property
    def confidence(self):
        return self.skeleton.confidence or None

    @confidence.setter
    def confidence(self, confidence):
        self.skeleton.confidence = confidence

    def __hash__(self):
        return id(self)

    def set_pose(self, frame_id, pose, angles, forward, clone_angles=True,
                 clone_forward=True):
        self.skeleton.set_pose(frame_id, pose)
        if angles is not None:
            self.skeleton.set_angles(frame_id, angles, clone=clone_angles)
        self.skeleton.set_forward(frame_id, forward, clone=clone_forward)

    def add_aux_info(self, key, info):
        """
        Other keys than '3d'.
        :param key: The top-level key in the output json
        :param info: The value of that key
        """
        assert key != 'frame_ids', "Times should be in skeleton"
        if key in self.aux_info:
            warnings.warn("aux_info with key %s already exists!" % key)
        self.aux_info[key] = info

    def get_object(self, obj_id):
        assert obj_id in self.objects, \
            "Does not have object with id %d, only: %s" % \
            (obj_id, self.objects.keys())
        return self.objects[obj_id]

    def get_obj_ids_for_label(self, label):
        return [oid for oid, o in self.objects.items() if o.label == label]

    def get_objects_for_label(self, label):
        oids = self.get_obj_ids_for_label(label)
        return [self.get_object(oid) for oid in oids]

    def has_object(self, obj_id):
        return obj_id in self.objects

    def add_object(self, obj_id, scene_obj, clone=True):
        """Adds an object to the scenelet.

        Arguments:
            obj_id (int):
                Unique id of the object inside the scenelet.
                Assigned automatically, if negative.
            scene_obj (stealth.logic.scenelet.SceneObj):
                Object to add.
            clone (bool):
                Deepcopy reference (True) or just save the reference (False).
        Returns:
            Reference to the object added.
        """

        if obj_id < 0:
            obj_id = len(self.objects)
        assert obj_id not in self.objects, \
            "Object with id %s already exists (name: %s)" % \
            (obj_id, self.objects[obj_id])
        if clone:
            self.objects[obj_id] = copy.deepcopy(scene_obj)
        else:
            self.objects[obj_id] = scene_obj

        return self.objects[obj_id]

    def set_transform(self, frame_id, transform):
        assert transform.shape[0] >= 3 and transform.shape[1] == 4, \
            "Need 3x4 or 4x4: %s" % transform.shape.__repr__()
        self.transforms[frame_id] = transform

    def get_transform(self, frame_id):
        return self.transforms[frame_id]

    def has_transform(self, frame_id):
        return frame_id in self.transforms

    def save(self, path, save_obj=False):
        """Save to disk.

        Args:
            path (str):
                Output path
            save_obj (bool):
                Save obj-s not just json for objects.
        """
        if path[-4:] == 'json':
            path = path[:-5]
        out = self.skeleton.to_json()
        assert 'frame_ids' in out, "Time should be in skeleton..."

        if self.name_scene is not None and len(self.name_scene):
            out['name_scene'] = self.name_scene
        if self.name_scenelet is not None and len(self.name_scenelet):
            out['name_scenelet'] = self.name_scenelet

        # aux_info
        for key, value in sorted(self.aux_info.items()):
            if key == 'frame_ids':
                raise DeprecationWarning("Time is in skeleton now")
            else:
                assert key not in out, \
                    "Key already added? %s" % key
                out[key] = value

        out_folder, stem = os.path.split(path)
        obj_folder = os.path.join(out_folder, stem + "_objects")
        if len(self.objects):
            if os.path.exists(obj_folder):
                shutil.rmtree(obj_folder)
            os.mkdir(obj_folder)

        obj_paths = {}
        for obj_id, scene_obj in sorted(self.objects.items()):
            added_paths = \
                scene_obj.save(
                  path_prefix=os.path.join(obj_folder,
                                           scene_obj.get_name(obj_id)),
                  save_obj=save_obj)
            for k, v in added_paths.items():
                if k in obj_paths:
                    obj_paths[k].extend(v)
                else:
                    obj_paths[k] = v
        out['scenelets'] = {}
        for k, v in sorted(obj_paths.items()):
            out['scenelets'][k] = \
                [
                    os.path.join(os.path.basename(obj_folder),
                                 os.path.split(p)[1])
                    for p in v
                ]

        j_transforms = {}
        for frame_id, transform in self.transforms.items():
            j_transforms[frame_id] = transform.tolist()
        out['scenelets']['transforms'] = j_transforms
        out['charness'] = float(self.charness)
        out['match_score'] = float(self.match_score)
        # if hasattr(self, 'confidence') and self.confidence is not None:
        #     out['confidence'] = self.confidence
        if hasattr(self, 'density') and self.density is not None:
            out['density'] = self.density
        if hasattr(self, 'charness_poses') and self.charness_poses is not None:
            out['charness_poses'] = {
                'frame_ids': list(self.charness_poses.keys()),
                'values': [float(v) for v in self.charness_poses.values()]
            }

        # write to disk
        json_path = path + ('.json' if path[-4:] != 'json' else '')
        with codecs.open(json_path, 'w', encoding='utf-8') as fout:
            if MyJsonEncode is None:
                json.dump(out, fout, sort_keys=True, indent=4)
            else:
                json.dump(out, fout, sort_keys=True, indent=4,
                          cls=MyJsonEncode)
        lg.info("Wrote to %s" % os.path.abspath(json_path))

    def to_mdict(self):
        out = self.skeleton.to_json()
        # aux_info
        for key, value in self.aux_info.items():
            assert key not in out, \
                "Key already added? %s" % key
            out[key] = value
        out['scenelets'] = {'obbs': []}
        for obj_id, scene_obj in self.objects.items():
            for part_id, part in scene_obj.parts.items():
                out['scenelets']['obbs'].append(part.obb.to_json(part_id))

        return out

    @classmethod
    def load(cls, path, no_obj=False):
        """

        Returns:
            s (stealth.logic.scenelet.Scenelet):
                Scenelet read from disk.
        """
        with open(path, 'r') as fin:
            data = json.load(fin)
        s = Scenelet()

        # 3d
        if '3d' in data:
            s.skeleton = Skeleton.from_json(data)
        else:
            sys.stderr.write("No skeleton in scenelet: %s" % path)

        # scenelets
        if 'scenelets' in data:
            if 'obbs' in data['scenelets'] and not no_obj:
                for obb_path in data['scenelets']['obbs']:
                    obb_folder, obj_name = os.path.split(obb_path)
                    obj_id, scene_obj_name, part_name = obj_name.split('_')
                    obj_id = int(obj_id)
                    part_name = part_name.split('.')[0]
                    obb_json_path = os.path.join(
                        os.path.split(path)[0],
                        obb_folder,
                        os.path.splitext(obj_name)[0] + ".json")
                    if os.path.exists(obb_json_path):
                        part = SceneObjPart(part_name)
                        part.obb, data_obb = Obb.load(obb_json_path,
                                                      return_data=True)
                        part_id = data_obb['part_id'] \
                            if 'part_id' in data_obb \
                            else -1
                        part.path = os.path.join(
                            os.path.split(path)[0], obb_path)
                        if s.has_object(obj_id):
                            s.get_object(obj_id).add_part(
                               part_id=part_id, label_or_part=part)
                        else:
                            scene_obj = SceneObj(scene_obj_name)
                            scene_obj.add_part(
                               part_id=part_id, label_or_part=part)
                            s.add_object(obj_id, scene_obj, clone=False)
                    else:
                        obb_obj_path = os.path.join(
                            os.path.split(path)[0], obb_path)
                        s.add_object(
                            obj_id,
                            SceneObject(
                                label=scene_obj_name,
                                name=part_name,
                                mesh=MeshOBJ(obb_obj_path, verbose=False)),
                            clone=False)
                    if 'transforms' in data['scenelets']:
                        try:
                            s.objects[obj_id].transform = \
                                np.asarray(
                                    data['scenelets']['transforms'][obj_name],
                                    dtype='f4')
                        except KeyError:
                            pass

            # TODO: read 'clouds' to parts

        for k, v in data.items():
            if k in Scenelet.__KEYS_SKELETON:
                continue
            s.add_aux_info(k, v)
        try:
            s.name_scene = data['name_scene']
        except KeyError:
            s.name_scene = os.path.basename(os.path.split(path)[-2])
            if '__' in s.name_scene:
                s.name_scene = s.name_scene.partition('__')[0]
            if s.name_scene.startswith('skel_'):
                s.name_scene = s.name_scene[5:]

        try:
            s.name_scenelet = data['name_scenelet']
        except KeyError:
            s.name_scenelet = os.path.splitext(os.path.basename(path))[0]
            if s.name_scenelet.startswith('skel_'):
                s.name_scenelet = s.name_scenelet[5:]

        if 'charness' in data:
            s.charness = data['charness']
        if 'match_score' in data:
            s.match_score = data['match_score']
        # if 'confidence' in data:
        #     s.confidence = data['confidence']

        # density
        try:
            s.density = data['density']
        except KeyError:
            pass

        # charness_poses
        try:
            j_dict = data['charness_poses']
            s.charness_poses = {
                frame_id: np.float32(value) for frame_id, value
                in zip(j_dict['frame_ids'], j_dict['values'])
            }
        except KeyError:
            pass

        return s

    def apply_transform(self, transform):
        self.skeleton.apply_transform(transform)
        for obj_id, scene_obj in self.objects.items():
            scene_obj.apply_transform(transform)
        if "preproc" not in self.aux_info:
            self.add_aux_info("preproc", {"transforms": []})
        self.aux_info["preproc"]["transforms"].append(transform.tolist())

    def get_ground_inv_transform(self):
        # get ground object obb rotation
        time_find_ground = time.clock()
        ground_part = None
        for scene_obj in self.objects.values():
            _part = scene_obj.get_part_by_name('floor')
            if _part:
                ground_part = _part
                break
        time_find_ground_end = time.clock()
        print("[Time] find_ground: %f" %
              (time_find_ground_end - time_find_ground))

        if ground_part:
            ground_transform = ground_part.obb.as_transform()
            ground_transform[:, 1], ground_transform[:, 2] = \
                -ground_transform[:, 2], -ground_transform[:, 1]
            # print("ground_transform:\n %s" % ground_transform)
            inv_transform = np.dot(rot_y(-np.pi / 2., (4, 4)),
                                   np.linalg.inv(ground_transform))
            inv_transform = np.linalg.inv(ground_transform)
        else:
            inv_transform = None
            raise RuntimeError("No ground!")

        return inv_transform

    def transform_to_local(self, ground_only=False):
        # get ground object obb rotation
        # time_find_ground = time.clock()
        # ground_part = None
        # for scene_obj in self.objects.values():
        #     _part = scene_obj.get_part_by_name('floor')
        #     if _part:
        #         ground_part = _part
        #         break
        # time_find_ground_end = time.clock()
        # print("[Time] find_ground: %f" %
        #       (time_find_ground_end - time_find_ground))

        # if ground_part:
        #     ground_transform = ground_part.obb.as_transform()
        #     ground_transform[:, 1], ground_transform[:, 2] = \
        #         -ground_transform[:, 2], -ground_transform[:, 1]
        #     inv_transform = np.dot(rot_y(-np.pi / 2., (4, 4)),
        #                            np.linalg.inv(ground_transform))
        #     inv_transform = np.linalg.inv(ground_transform)
        #     print("inv_transform:\n %s" % inv_transform)
        #
        #     self.apply_transform(inv_transform)
        inv_transform = self.get_ground_inv_transform()
        if inv_transform is not None:
            self.apply_transform(inv_transform)

        frames = list(self.skeleton.get_frames())
        if not ground_only:
            mid_frame = frames[int(len(frames)/2)]
            # move to skeleton centroid (x and z, not y)
            skel_transform = self.skeleton.get_transform(mid_frame)
            # translate to centered pelvis along xz
            tr0 = \
                translation((-skel_transform[0, 3], 0., -skel_transform[2, 3]))
            # orient to unit_x
            unit_x = np.asarray([1., 0., 0.], dtype=np.float32)
            forward = self.skeleton.get_forward(mid_frame, estimate_ok=False, k=2)
            forward[1] = 0.
            angle = np.arccos(forward.dot(unit_x))
            tr1a = rot_y(angle)
            tr1b = rot_y(-angle)
            forward_a = np.dot(tr1a[:3, :3], forward)
            forward_b = np.dot(tr1b[:3, :3], forward)
            tr1 = tr1a \
                if np.dot(forward_a, unit_x) > np.dot(forward_b, unit_x) \
                else tr1b

            # apply
            # self.apply_transform(skel_transform)
            self.apply_transform(np.dot(tr1, tr0))
            self.aux_info['forward'] = forward.tolist()
            # print("applying:\n%s" % tr0)
            # self.apply_transform(tr0)
            # skel_transform2 = self.skeleton.get_transform(mid_frame)
            # print("skel_transform2:\n%s" % skel_transform2)
            # print("skel_transform:\n%s" % skel_transform)


            # (rx, ry, rz) = rot_to_euler(skel_transform)
            # skel_transform[:3, :3] = rot_y(ry)
            # self.transform(np.linalg.inv(skel_transform))
            # print("transform: %s" % transform)
            # inv_transform = np.linalg.inv(transform)
            # print("inv_transform: %s" % inv_transform)
            # self.skeleton.transform(inv_transform)
            # for obj_id, scene_obj in self.objects.items():
            #     scene_obj.transform(inv_transform)

        # start at frame 0:
        self.skeleton.move_in_time(-frames[0])

    def __repr__(self):
        frames = self.skeleton.get_frames()
        return "Scenelet(%s, %s, %d..%d)" % \
               (self.name_scene, self.name_scenelet,
                frames[0] if frames else 0,
                frames[-1] if frames else 0)

    def get_labels(self, ignore=set(), group=True):
        """Returns an occurrence list of object categories.

        Args:
            ignore (set): Categories to ignore and not return.
            group (bool): Do group to an occurrence dictionary (Counter).
        """
        cat = [o.label
               for o in self.objects.values()
               if o.label not in ignore]
        if group:
            return dict(Counter(cat).items())
        else:
            return cat

    def __eq__(self, other):
        assert self.name_scenelet is not None \
            and self.name_scene is not None, "None?"
        return self.name_scenelet == other.name_scenelet \
            and self.name_scene == other.name_scene

    def get_time(self, frame_id):
        """Get's fractional time instead of frame_id.

        Args:
             frame_id (int):
        Returns:
            time (float):
        """
        assert isinstance(frame_id, (int, np.int64)), \
            "Expected int, not %s" % type(frame_id)
        assert 'frame_ids' not in self.aux_info, \
            "Time should be in skeleton"

        try:
            return self.skeleton.get_time(frame_id=frame_id)
        except KeyError:
            return frame_id

    def center_time(self):
        """Move frame_ids so that they match the center of the times
        (frame_ids in aux_info).

        Returns:
            _: (None)
        """
        assert 'frame_ids' not in self.aux_info, "Time should be in skeleton"

        self.skeleton.center_time()

    @staticmethod
    def to_old_name(name):
        """

        Args:
            name (str):
        Returns:
            parts (Tuple[str, str]):
        """
        if '__' in name:
            parts = name.split('__')
            return parts[0], "skel_%s" % parts[1]
        else:
            parts = name.split('_scenelet')
            return parts[0], "skel_scenelet%s" % parts[1]

    @staticmethod
    def get_scene_name_from_recording_name(name_recording):
        """Get scene name from recording name.

        Args:
            name_recording (str):
        Returns:
            scene_name (str):
        """
        return name_recording.partition('_')[0]

    def get_rectangles(self):
        """Return an ordered list of bounding rectangles on the
        x-z plane.
        """
        return [ob.get_obb().corners_3d_lower()[:, [0, 2]]
                for ob in self.objects.values()]

    def get_charness_pose(self, frame_id):
        """

        Args:
            frame_id (int):
        Returns:
            charness (np.float32):
        """
        assert isinstance(frame_id, int), \
            "Expected int, not %s" % type(frame_id)
        try:
            return self.charness_poses[frame_id]
        except KeyError:
            return np.float32(0.)

    def set_charness_pose(self, frame_id, charness):
        """Per-frame charness getter. Defaults to 0, if no entry.

        Args:
            frame_id (int):
                Integer key for the pose.
            charness (np.float32):
                Characteristicness of pose, usually [0., 1.].
        Returns:
            _ (None):
        """
        assert isinstance(frame_id, int), \
            "Expected int, not %s" % type(frame_id)
        assert isinstance(charness, (float, np.float32)), \
            "Expected float, not %s" % type(charness)
        if self.charness_poses is None:
            self.charness_poses = {frame_id: np.float32(charness)}
        else:
            self.charness_poses[frame_id] = np.float32(charness)


def read_scenelets(scenelet_dir, filter_lambda=None, transform=None,
                   skel_only=True):
    """Recursively reads a directory containing scenelets (json files).

    Args:
        scenelet_dir (str):
            Path to scan for scenelets.
        filter_lambda (Callable[[Scenelet], bool]):
            Takes a scenelet, and returns False for reject, True for keep.
        transform (np.ndarray):
            Transformation to apply to scenelets.
        skel_only (bool):
            Read only files starting with 'skel'.

    Returns:
        py_scenes (Dict[str, Dict[str, Scenelet]]):
            Scenelets keyed by their scene name and their recording name.
    """
    py_scenes = {}
    for parent, dirs, files in os.walk(scenelet_dir):
        name_scene = None
        for f in [f for f in files if f.endswith('.json')]:
            if skel_only and not f.startswith('skel'):
                continue
            if '__' in f:
                name_parts = f.split('__')
                # lg.debug("name_parts: %s" % name_parts)
                name_scene = name_parts[0]
                if name_scene.startswith('skel_'):
                    name_scene = name_scene[5:]
                name_sclt = "skel_%s" % os.path.splitext(name_parts[1])[0]
            else:
                if not name_scene:
                    name_scene = os.path.basename(os.path.split(parent)[-1])
                    if not len(name_scene):
                        name_scene = \
                            os.path.basename(os.path.split(parent[:-1])[-1])
                    py_scenes[name_scene] = {}
                name_sclt = os.path.splitext(f)[0]
                # lg.debug("name scene: %s, name_scenelet: %s"
                #               % (name_scene, name_sclt))
            j_path = os.path.join(parent, f)
            sclt = Scenelet.load(j_path)

            # save to output
            sclt.name_scene = name_scene
            sclt.name_scenelet = name_sclt

            if filter_lambda and not filter_lambda(sclt):
                # lg.debug("filtering %s" % sclt)
                continue

            if transform is not None:
                sclt.apply_transform(transform)

            try:
                while name_sclt in py_scenes[name_scene]:
                    lg.warning(
                        "Modifying scenelet map key in order not to "
                        "overwrite %s in %s, (keys; %s)"
                        % (name_sclt, name_scene,
                           sorted(list(py_scenes[name_scene].keys()))))
                    name_sclt = "%s_" % name_sclt
                py_scenes[name_scene][name_sclt] = sclt
            except KeyError:
                py_scenes[name_scene] = {name_sclt: sclt}
            # lg.info("Added %s" % sclt)
            del name_sclt

    return py_scenes


def get_scenelets(py_scenes):
    """Traverses the [scene][scenelet] structure and returns a generator.

    Args:
        py_scenes (Dict[str, Dict[str, Scenelet]]):
            Scenelets grouped by scene name and recording name.
    Returns (Iterable[Tuple[str, str, Scenelet]]):
        (name_scene, name_sclt, Scenelet)
    """

    return ((name_scene, name_sclt, sclt)
            for name_scene, sclts in py_scenes.items()
            for name_sclt, sclt in sclts.items())

# try:
#     from stealth.scenelet_fit.unknowns_manager \
#         import um_set_n_objects, um_set_pos, um_set_rot, \
#         um_get_x_length, um_set_category_id, um_set_model_id
# except ImportError as e:
#     print("\t[scenelet.py] Could not load unknowns_manager.pyx: %s\n\tThis is "
#           "ok, exception caught." % e)
#
#
# def prepare_rects(scenelet, labels_to_lin_ids, models, update_scene):
#     """Take world-space scenelet models with their transforms
#     and move them back to local-space
#     """
#     x = np.zeros(
#         shape=(um_get_x_length(len(scenelet.objects)),),
#         dtype='f4')
#     um_set_n_objects(x=x, n_objects=len(scenelet.objects))
#     rects = []
#     for unk_i, obj in enumerate(scenelet.objects.values()):
#         t = obj.transform.copy()
#
#         theta = obj.get_angle(positive_only=True)  # - HALF_PI_32
#
#         #
#         # set x vector
#         #
#
#         # pos
#         um_set_pos(
#            x=x, object_linear_id=unk_i, pos=t[[0, 2], 3])
#         # rot
#         um_set_rot(
#             x=x, object_linear_id=unk_i, rot=theta)
#         # category id
#         cat_id = labels_to_lin_ids[obj.label]
#         um_set_category_id(
#             x=x, object_linear_id=unk_i, cat_id=cat_id)
#         # model_id
#         model_id, model = next((mi, m)
#                                for mi, m in enumerate(models[cat_id])
#                                if m.name == obj.name)
#         um_set_model_id(
#             x=x, object_linear_id=unk_i, model_id=model_id)
#
#         #
#         # Untransform scene
#         #
#
#         if update_scene:
#             inv_transform = np.linalg.inv(t)
#             obj.apply_transform(inv_transform)
#
#             # estimate obb in local space
#             obj.get_obb()
#
#             # transform to forward = (1, 0), pos_y = original pos_y
#             t[:3, :3] = rot_y(-HALF_PI_32)[:3, :3]
#             t[0, 3] = np.float32(0.)
#             t[2, 3] = np.float32(0.)
#             obj.apply_transform(t)
#
#             # save rectangle
#             corners = obj.get_obb().corners_3d_lower()[:, [0, 2]]
#             rects.append(np.append(corners, [corners[0, :]], axis=0))
#             rects[-1].flags.writeable = False
#
#     if update_scene:
#         return rects, x
#     else:
#         return x
