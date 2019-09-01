import codecs
import copy
import json
import sys
import bisect

import numpy as np
from builtins import range
if not sys.version_info[0] < 3:
    from functools import lru_cache
    from typing import Dict
else:
    from repoze.lru import lru_cache

import imapper.logic.geometry as gm
from imapper.logic.joints import Joint
from imapper.logic.scene_object import Obb
from imapper.util.stealth_logging import lg


class ForwardEstimationError(RuntimeError):
    def __init__(self, msg):
        super(ForwardEstimationError, self).__init__(msg)


class NActorsError(RuntimeError):
    pass


class Skeleton(object):
    N_JOINTS = Joint.get_num_joints()  # type: int
    """Number of joints in skeleton."""
    DIM = 3  # type: int
    """Dimensionality of a joint position."""

    def __init__(self, frames_mod=None, n_actors=None, min_frame_id=None):
        self._poses = None
        """Frames x DIM x N_JOINTS array of SPARSE skeleton positions"""
        self._frame_ids = {}
        """frame_id => linear id, where _poses[linear_id, :, :] 
        is the pose at time frame_id"""
        self._frames_cache = None
        """Maintains a sorted version of _frame_ids.keys()."""
        self._visibility = dict()
        """Flags for joint visibility. 
           Type: Dict[int, Dict[Joint, bool]]
        """
        self._confidence = dict()
        """Joint detection confidence.
           Type: Dict[int, Dict[Joint, float]]
        """
        self._forwards = dict()
        """Interpolated forward vectors"""
        self._angles = dict()
        """Interpolated descriptors"""

        self._changed = False  # type: bool
        """Dirty flag."""
        self._bone_obbs = {}
        """Unused"""

        self._times = {}
        """Non-integer time indexed by integer frame_id."""

        self._frames_mod = None  # type: int
        """Maximum number of frames for one actor, e.g. the number of 
        images in the video."""
        self._min_frame_id = None  # type: int
        """Fisst image id in video"""
        self._n_actors = None  # type: int
        """Number of actors stored in skeleton."""

        self.set_n_actors(frames_mod=frames_mod, min_frame_id=min_frame_id,
                          n_actors=n_actors)

        self._confidence_normalized = False  # type: bool
        """Whether the confidence values are already normalized.
            Type: bool
        """

    def set_n_actors(self, frames_mod, min_frame_id, n_actors):
        """Setup multi-actor information.

        Args:
            frames_mod (int):
                Length of video.
            min_frame_id (int):
                First frame of video.
            n_actors (int):
                Number of actors in skeleton
        """
        assert (frames_mod is None) == (n_actors is None) \
               == (min_frame_id is None), \
            "Need both frames_mod and n_actors argument"
        if self.n_actors is not None and n_actors is not None \
          and self.n_actors < n_actors and len(self._frame_ids):
            raise NActorsError("Can't decrease the number of actors, "
                               "call 'clear_poses' first")
        self._frames_mod = frames_mod
        self._min_frame_id = min_frame_id
        self._n_actors = n_actors

    @classmethod
    def from_ndarray(cls, poses):
        assert poses.shape[1] == cls.DIM, \
            "Wrong shape: %s" % repr(poses.shape)
        assert poses.shape[2] == cls.N_JOINTS, \
            "Wrong shape: %s" % repr(poses.shape)

        skeleton = Skeleton()
        skeleton._poses = poses
        skeleton._frame_ids = \
            {lin_id: lin_id for lin_id in range(poses.shape[0])}
        # skeleton._times = \
        #     {lin_id: lin_id for lin_id in range(poses.shape[0])}
        skeleton.set_changed(True)
        return skeleton

    @classmethod
    def check_pose_shape(cls, pose):
        assert pose.shape[0] == Skeleton.DIM, \
            "Assumed vertices in cols: %s" % repr(pose.shape)
        assert pose.shape[1] == Skeleton.N_JOINTS, \
            "Assumed %d joints: %s" % (Skeleton.N_JOINTS, pose.shape)
        return True

    @classmethod
    def get_joint_3d_(cls, joint_id, pose):
        assert isinstance(joint_id, int), "Expected int, %s" % type(joint_id)
        cls.check_pose_shape(pose)
        assert joint_id < pose.shape[1], \
            "Not enough joins: %d >= %d" % (joint_id, pose.shape[1])
        return pose[:, joint_id]

    @classmethod
    def set_joint_3d_(cls, joint_id, vertex, pose):
        assert pose.dtype == np.float32, "Wrong type: %s" % pose.dtype
        assert isinstance(joint_id, int), "Expected int, %s" % type(joint_id)
        cls.check_pose_shape(pose)
        assert joint_id < pose.shape[1], \
            "Not enough joins: %d >= %d" % (joint_id, pose.shape[1])
        pose[:, joint_id] = vertex

        return pose

    def move_in_time(self, frame_offset):
        """Adds time_offset to all frameIds.

        Args:
            frame_offset (int): Number of frames to move by.
        Returns:
            _ (None):
        """

        if frame_offset == 0:
            return None

        self._frame_ids = \
            dict((key + frame_offset, value)
                 for (key, value) in self._frame_ids.items())
        assert not any(f < 0 for f in self.get_frames()), \
            "Negative frame_id: %s" % self.get_frames()
        self._frames_cache = None

        self._visibility = \
            dict((key + frame_offset, value)
                 for (key, value) in self._visibility.items())
        self._confidence = \
            dict((key + frame_offset, value)
                 for (key, value) in self._confidence.items())
        self._forwards = \
            dict((key + frame_offset, value)
                 for (key, value) in self._forwards.items())
        self._angles = \
            dict((key + frame_offset, value)
                 for (key, value) in self._angles.items())
        self._times = \
            dict((key + frame_offset, value)
                 for key, value in self._times.items())
        self.set_changed(True)

    def copy_frame_from(self, frame_id, other,
                        may_overwrite=False, dest_frame_id=None):
        """Copies the full information about a pose from another
        skeleton. Maintaining this ensures, that we don't forget to
        update a field.

        Args:
            frame_id (int):
            other (Skeleton):
            may_overwrite (bool):
            dest_frame_id (int):
                Frame_id in this skeleton (self). Set to frame_id, if None.
        Returns:
            _ (None):
        """

        if dest_frame_id is None:
            dest_frame_id = frame_id

        if not may_overwrite and dest_frame_id in self._frame_ids:
            raise RuntimeError("Won't overwrite frame %d" % dest_frame_id)

        # _frame_ids, _poses, _times
        self.set_pose(frame_id=dest_frame_id,
                      pose=other.get_pose(frame_id=frame_id),
                      time=other.get_time(frame_id=frame_id))

        # _visibility
        for frame_id in other._visibility:
            for joint in other._visibility[frame_id]:
                self.set_visible(
                    frame_id=dest_frame_id, joint=joint,
                    visible=other.is_visible(frame_id, joint)
                )

        if hasattr(other, '_confidence'):
            for frame_id in other._confidence:
                for joint in other._confidence[frame_id]:
                    self.set_confidence(
                        frame_id=dest_frame_id, joint=joint,
                        confidence=other.get_confidence(frame_id, joint))

        # _forwards
        try:
            self.set_forward(frame_id=dest_frame_id,
                             forward=other.get_forward(frame_id=frame_id,
                                                       estimate_ok=False),
                             clone=True)
        except KeyError:
            pass

        # _angles
        try:
            self.set_angles(frame_id=dest_frame_id,
                            angles=other._angles[frame_id],
                            clone=True)
        except KeyError:
            pass

    def get_joint_3d(self, joint_id, frame_id):
        assert isinstance(frame_id, (int, np.int64)), \
            "Expected int, %s" % type(frame_id)
        assert isinstance(joint_id, int), \
            "Expected int, %s" % type(joint_id)
        lin_id = self._frame_ids[frame_id]
        return self._poses[lin_id, :, joint_id]

    def get_centroid_3d(self, frame_id):
        lin_id = self._frame_ids[frame_id]
        return np.mean(self._poses[lin_id, :, :], axis=-1)

    def get_joint_3d_all(self, joint_id):
        assert isinstance(joint_id, int), "Expected int, %s" % type(joint_id)
        return self._poses[:, :, joint_id]

    def get_joint_3d_by_lin_id(self, joint_id, lin_id):
        assert isinstance(joint_id, int), "Expected int, %s" % type(joint_id)
        assert isinstance(lin_id, int), "Expected int, %s" % type(lin_id)
        return self._poses[lin_id, :, joint_id]

    def set_joint_3d(self, joint_id, frame_id, vertex):
        assert isinstance(joint_id, int), "Expected int, %s" % type(joint_id)
        assert isinstance(frame_id, int), "Expected int, %s" % type(frame_id)
        lin_id = self._frame_ids[frame_id]
        self._poses[lin_id, :, joint_id] = vertex.astype('f4')
        self.set_changed(True)

    def get_col_ids(self):
        """Returns the column ids of the sorted frame_ids"""
        return [self._frame_ids[f] for f in self.get_frames()]

    def set_pose(self, frame_id, pose, time=None):
        """Stores a 3D pose at 'frame' [and 'time'].

        Args:
            frame_id (int):
                Frame id in video.
            pose (np.ndarray): (Skeleton.DIM, Skeleton.N_JOINTS)
                Will never copy.
            time (float):
                Continuous time in video.
        """
        assert isinstance(frame_id, (int, np.int64)), \
            "Expected int, %s" % type(frame_id)
        if pose.dtype != np.float32:
            pose = pose.astype(np.float32)
        np_pose = None

        if isinstance(pose, np.ndarray):
            Skeleton.check_pose_shape(pose)
            np_pose = pose
            # print("np_pose.shape: %s" % np_pose.shape.__repr__())
        elif isinstance(pose, dict):
            np_pose = \
                np.zeros((Skeleton.DIM, Skeleton.N_JOINTS), dtype=np.float32)
            for joint_id, vertex in pose.items():
                # self._poses[lin_id, :, joint_id] = vertex
                np_pose[:, joint_id] = vertex.astype(np.float32)
        else:
            print("Could not parse, type is %s" % type(pose))
            raise RuntimeError("could not parse %s" % pose)
        assert np_pose.dtype == np.float32, "Wrong type: %s" % np_pose.dtype

        if not len(self._frame_ids.keys()):  # if first frame ever
            assert self._poses is None or len(self._poses) == 0, \
                "Not none? %s" % self._poses
            self._poses = np.expand_dims(np_pose, axis=0)
            assert self._poses is not np_pose, "Need deepcopy!"
            self._frame_ids[frame_id] = 0
        elif frame_id not in self._frame_ids:
            num_el = len(self._frame_ids.keys())
            col_id = next(
                (col for col, _frame_id in enumerate(self.get_frames())
                 if _frame_id > frame_id), num_el)

            # append or insert
            if col_id == num_el:
                self._poses = \
                    np.append(self._poses,
                              np.expand_dims(np_pose, axis=0),
                              axis=0)
            else:
                self._poses = np.insert(self._poses, col_id, np_pose, axis=0)

            # move all cols one forward
            for _frame_id, _col_id in self._frame_ids.items():
                if _col_id >= col_id and col_id < num_el:
                    self._frame_ids[_frame_id] += 1

            self._frame_ids[frame_id] = col_id  # col_id was upper bound
        else:
            lin_id = self._frame_ids[frame_id]
            self._poses[lin_id, :, :] = np_pose

        if time is not None:
            self._times[frame_id] = np.float32(time)
        else:
            assert frame_id in self._times or not len(self._times), \
                "Please provide time to set_pose, if we are using it."

        assert self._poses.dtype == np.float32, \
            "Wrong type: %s" % self._poses.dtype

        self.set_changed(True)
        self._frames_cache = None

    def set_angles(self, frame_id, angles, clone):
        """Stores a descriptor for an integer time.

        Args:
            frame_id (int):
                The frame_id of the angles.
            angles (np.ndarray):
                A column vector of angles.
            clone (bool):
                Deep copy, if true.
        Returns:
            None
        """
        assert isinstance(frame_id, int), "Expected int, %s" % type(frame_id)
        assert isinstance(angles, np.ndarray), "wrong type: %s" % type(angles)
        assert angles.shape == (14, ), "Wrong shape: %s" % repr(angles.shape)
        assert frame_id in self._frame_ids, \
            "Adding angles to non-existent pose?"
        if clone:
            self._angles[frame_id] = angles.copy()
        else:
            self._angles[frame_id] = angles

    def set_forward(self, frame_id, forward, clone):
        """

        Args:
            frame_id (int):
            forward (np.ndarray):
            clone (bool):
        Returns:
            None
        """
        assert isinstance(frame_id, int), "Expected int, %s" % type(frame_id)
        assert len(forward) == 3, "Expected 3d vector: %s" % forward
        assert frame_id in self._frame_ids, \
            "Adding angles to non-existent pose?"
        self._forwards[frame_id] = forward.astype(np.float32, copy=clone)

    def set_time(self, frame_id, time):
        assert isinstance(frame_id, int), "Expected int, %s" % type(frame_id)
        assert frame_id in self._frame_ids, \
            "Adding angles to non-existent pose?"
        self._times[frame_id] = time

    def remove_pose(self, frame_id):
        assert isinstance(frame_id, (int, np.int32, np.int64)), \
            "Expected int, %s" % type(frame_id)
        assert frame_id in self._frame_ids, \
            "No such frame: %s" % frame_id
        lin_id = self._frame_ids[frame_id]
        del self._frame_ids[frame_id]
        try:
            del self._times[frame_id]
        except KeyError:
            if len(self._times):
                sys.stderr.write("We have times, but not for this frame?\n")

        # print("frame_id for lin_id %d was %d" % (lin_id, frame_id))
        self._frame_ids = dict((k, v if k < frame_id else v-1)
                               for k, v in self._frame_ids.items())
        self._visibility.pop(frame_id, None)
        self._confidence.pop(frame_id, None)
        # print(self._poses[lin_id, :, :])
        # print("shape: %s" % repr(self._poses.shape))
        self._poses = np.delete(self._poses, lin_id, axis=0)
        if self._forwards is not None:
            self._forwards.pop(frame_id, None)
        if self._angles is not None:
            self._angles.pop(frame_id, None)

        self._changed = True
        self._frames_cache = None

        # print("shape: %s" % repr(self._poses.shape))
        # print(self._poses[lin_id, :, :])

    def clear_poses(self):
        """Clears frame_id related information, but keep additional info."""
        self._poses = None
        self._frame_ids = dict()
        self._frames_cache = None
        self._visibility = dict()
        self._confidence = dict()
        self._forwards = dict()
        self._angles = dict()
        self._changed = True
        self._bone_obbs = dict()
        self._times = {}

    def get_pose(self, frame_id):
        """A 3D pose at frame 'frame_id'.

        Args:
            frame_id (int): Frame ID.
        Returns:
            pose (np.ndarray): (Skeleton.DIM, Skeleton.N_JOINTS)
                Pose at time 'frame_id'.
        """

        assert isinstance(frame_id, (int, np.int32, np.int64)), \
            "Expected int, %s" % type(frame_id)
        lin_id = self._frame_ids[frame_id]
        return self._poses[lin_id, :, :]

    def get_time(self, frame_id):
        """Get continuous time in video.

        Args:
            frame_id (int):
                Index of image in video.
        Returns:
            time (float):
                Time in video.
        """
        assert isinstance(frame_id, (int, np.int64)), \
            "Expected int, %s" % type(frame_id)
        return self._times[frame_id]

    def get_last_time(self):
        if not len(self._frame_ids):
            raise KeyError("Don't have any frames")
        return self.get_time(frame_id=max(self._frame_ids.keys()))

    def has_time(self, frame_id):
        assert isinstance(frame_id, (int, np.int64)), \
            "Expected int, %s" % type(frame_id)
        return frame_id in self._times

    def find_time(self, time):
        """Finds the frame_id that has closest time to the query time."""
        assert not isinstance(time, int), "Expected float, %s" % type(time)
        frame_id = min(((frame_id, t) for frame_id, t in self._times.items()),
                       key=lambda e: abs(time - e[1]))[0]
        lg.debug("frame_id: %d, time: %g, query time: %g"
                 % (frame_id, self.get_time(frame_id), time))
        assert abs(time - self.get_time(frame_id)) < 0.5, \
            "no: %g" % abs(time - self.get_time(frame_id))
        return frame_id

    @property
    def poses(self):
        return self._poses

    @poses.setter
    def poses(self, poses):
        self._poses = poses

    def has_pose(self, frame_id):
        return bool(frame_id in self._frame_ids)

    def to_json(self):
        """Prepare for being saved to disk."""
        out = {'3d': {}}
        # poses
        for frame_id in self.get_frames():
            pose = self.get_pose(frame_id)
            j_pose = {}
            for joint_id in range(pose.shape[1]):
                j_pose["%d" % joint_id] = \
                    pose[:, joint_id].astype(float).tolist()
            out['3d']["%05d" % frame_id] = j_pose
        if len(self._visibility):
            keys_srtd = sorted(self._visibility.keys())
            assert isinstance(keys_srtd, list)
            out['visibility'] = {
                'frame_ids': keys_srtd,
                'values': [{int(j): v for j, v in self._visibility[k].items()}
                           for k in keys_srtd]
            }
        if len(self._confidence):
            keys_srtd = sorted(self._confidence.keys())
            assert isinstance(keys_srtd, list)
            out['confidence'] = {
                'frame_ids': keys_srtd,
                'values': [{int(j): v.tolist() if (isinstance(v, np.ndarray) and v.size == 1) else v
                           for j, v in self._confidence[k].items()}
                           for k in keys_srtd]
            }
        if self._forwards is not None and len(self._forwards):
            out['forwards'] = \
                dict((frame_id_, fw_.astype(float).tolist())
                     for frame_id_, fw_ in sorted(self._forwards.items()))
        if self._angles is not None and len(self._angles):
            out['angles'] = \
                dict((frame_id_, angles_.astype(float).tolist())
                     for frame_id_, angles_ in sorted(self._angles.items()))

        out['frame_ids'] = [float(self._times[k])
                            for k in sorted(self._times)] \
            if len(self._times) \
            else list(sorted(self._frame_ids.keys()))

        if self._frames_mod is not None:
            out['actors'] = {
                'frames_mod': self._frames_mod,
                'n_actors': self.n_actors,
                'min_frame_id': self._min_frame_id
            }

        out['confidence_normalized'] = self.is_confidence_normalized() \
            if hasattr(self, '_confidence_normalized') \
            else False

        return out

    @classmethod
    def from_json(cls, data):
        """Deserializes a skeleton read from disk.

        Args:
            data (dict):
                Json data read from disk.
        Returns:
            skeleton (Skeleton):
                Parsed Skeleton object.
        """

        # TODO: speed up by allocate at once
        assert '3d' in data, \
            "Assumed getting the whole data: %s" % data
        s = Skeleton()
        for k, v in sorted(data['3d'].items()):
            frame_id = int(k)
            pose = np.zeros(shape=(len(v[list(v.keys())[0]]), len(v)),
                            dtype=np.float32)
            assert pose.shape == (cls.DIM, cls.N_JOINTS), \
                "Wrong pose.shape: %s" % pose.shape.__repr__()
            for joint, pos in v.items():
                pose[:, int(joint)] = pos
            s.set_pose(frame_id, pose)
        if 'visibility' in data:
            # s._visibility = dict((int(str(k)), bool(v))
            #                      for k, v in data['visibility'].items())
            j_dict = data['visibility']
            s._visibility = {
                frame_id: {int(j): v for j, v in value.items()}
                for frame_id, value
                in zip(j_dict['frame_ids'], j_dict['values'])
            }
        if 'confidence' in data:
            j_dict = data['confidence']
            if isinstance(j_dict, dict):
                s._confidence = {
                    frame_id: {
                        int(j): v
                        for j, v in value.items()}
                    for frame_id, value
                    in zip(j_dict['frame_ids'], j_dict['values'])
                }
            else:
                lg.error("[Scenelet.from_json] Ignoring \'confidence\' field.")

        if 'confidence_normalized' in data:
            s._confidence_normalized = data['confidence_normalized']
            assert isinstance(s._confidence_normalized, bool), \
                "wrong type: %s" % type(s._confidence_normalized)
        else:
            s._confidence_normalized = False
            lg.warning("Assuming unnormalized confidence...")

        if 'forwards' in data:
            s._forwards = dict((int(str(k)), np.asarray(v))
                               for k, v in data['forwards'].items())
        if 'angles' in data:
            s._angles = dict((int(str(k)), np.asarray(v))
                             for k, v in data['angles'].items())
        if 'frame_ids' in data:
            s._times = {int(frame_id): data['frame_ids'][i]
                        for i, frame_id in enumerate(sorted(data['3d'].keys()))}
            print(s)
            if s.poses is not None:
                assert len(s._times) == s.poses.shape[0], \
                    "Wrong times...%s %s" % (len(s._times), s.poses.shape)
        if 'actors' in data:
            actors = data['actors']
            s._frames_mod = actors['frames_mod']
            s._n_actors = actors['n_actors']
            s._min_frame_id = actors['min_frame_id']
        return s

    @classmethod
    def load(cls, path):
        """Reads skeleton file from disk.

        Args:
            path (str):
                Path to skeleton.
        Returns:
            skeleton (Skeleton):
                Parsed skeleton.
        """
        with codecs.open(path, 'r', encoding='utf-8') as fin:
            return cls.from_json(json.load(fin))

    def get_lin_id_for_frame_id(self, frame_id):
        """

        Args:
            frame_id (int):
                Frame id.

        Returns:
            lin_id (int):
                Linear id of frame_id.
        """
        return self._frame_ids[frame_id]

    def get_frame_id_for_lin_id(self, lin_id):
        return next((frame_id for frame_id, lin_id_ in self._frame_ids.items()
                     if lin_id_ == lin_id), None)

    def get_frames(self):
        """Get frame_ids sorted."""
        if not self._frames_cache:
            self._frames_cache = sorted(self._frame_ids)
        return self._frames_cache

    def get_frames_mod(self):
        """Modulates frame_ids for more-actor skeletons."""
        frame_ids = self.get_frames()
        return [self.mod_frame_id(frame_id=f) for f in frame_ids]

    @staticmethod
    def unmod_frame_id(frame_id, actor_id, frames_mod):
        """Converts an image_id to an extended linear frame_id.

        Args:
            frame_id (int):
                The frame_id in question.
            mod (int):
                The number of images.
            min_frame_id (int):
                The frame_id of the first image.
        Returns:
            frame_id_mod (int):
                The frame_id that has an image equivalent,
                e.g. "color_%05d.jpg" % frame_id_mod exists.
        """

        if frames_mod is None:
            assert actor_id == 0, \
                "need 1 actor when no frames_mod, actor_id={}".format(actor_id)
            return frame_id
        else:
            return frame_id + frames_mod * actor_id

    def mod_frame_id(self, frame_id):
        """Converts an extended frame_id to one that is guaranteed to
        have an image correspondence.

        Args:
            frame_id (int):
                The frame_id in question.
            mod (int):
                The number of images.
            min_frame_id (int):
                The frame_id of the first image.
        Returns:
            frame_id_mod (int):
                The frame_id that has an image equivalent,
                e.g. "color_%05d.jpg" % frame_id_mod exists.
        """
        if self.n_actors == 1:
            return frame_id
        min_frame_id = self._min_frame_id
        assert frame_id >= min_frame_id
        assert self.get_actor_id(frame_id) < self.n_actors
        return (frame_id - min_frame_id) % self._frames_mod \
               + min_frame_id

    def get_actor_id(self, frame_id):
        """
        Args:
            frame_id (int): The frame id.
        Returns:
             actor_id (int): The actor id.
        """
        if self._min_frame_id is None:
            return 0

        min_frame_id = self._min_frame_id
        return (frame_id - min_frame_id) // self._frames_mod

    def get_actor_last_frame(self, actor_id):
        """Finds the largest frame_id of actor.

        Args:
            actor_id (int):
                The zero-indexed id of the actor.
        Returns:
            frame_id (int):
                The last linear frame_id that belongs to actor.
        """
        if actor_id >= self.n_actors:
            raise RuntimeError("Actor id out of bounds: %d >= %d"
                               % (actor_id, self.n_actors))

        frames = self.get_frames()
        if actor_id == self.n_actors - 1:
            return frames[-1]
        first = self.unmod_frame_id(frame_id=self._min_frame_id,
                                    actor_id=actor_id + 1,
                                    frames_mod=self._frames_mod) - 1
        while first not in frames:
            first -= 1
        return first

    def get_actor_first_frame(self, actor_id):
        """Finds the smallest frame_id of actor.

        Args:
            actor_id (int):
                The zero-indexed id of the actor.
        Returns:
            frame_id (int):
                The first linear frame_id that belongs to actor.
        """
        if actor_id >= self.n_actors:
            raise RuntimeError("Actor id out of bounds: %d >= %d"
                               % (actor_id, self.n_actors))

        frames = self.get_frames()
        if actor_id == 0:
            return frames[0]
        first = self.unmod_frame_id(frame_id=self._min_frame_id,
                                    actor_id=actor_id,
                                    frames_mod=self._frames_mod)
        while first not in frames and first <= frames[-1]:
            first += 1
        if first > frames[-1]:
            lg.error("No frames for actor %d" % actor_id)
            return False
        return first

    def get_frames_min_max(self):
        frames = self.get_frames()
        assert len(frames), "No frames"
        return frames[0], frames[-1]

    def get_transform(self, frame_id):
        assert self.has_pose(frame_id), \
            "Does not have frame_id: %d" % frame_id
        rHip = self.get_joint_3d(Joint.RHIP, frame_id=frame_id)
        lHip = self.get_joint_3d(Joint.LHIP, frame_id=frame_id)
        rSho = self.get_joint_3d(Joint.RSHO, frame_id=frame_id)
        lSho = self.get_joint_3d(Joint.LSHO, frame_id=frame_id)
        pelv = self.get_joint_3d(Joint.PELV, frame_id=frame_id)
        thrx = self.get_joint_3d(Joint.THRX, frame_id=frame_id)
        tr = np.eye(N=4, M=4, dtype=np.float32)
        tr[:3, 0] = gm.normalized(gm.normalized(lHip - rHip)
                                  + gm.normalized(lHip - pelv)
                                  + gm.normalized(pelv - rHip)
                                  + gm.normalized(lSho - rSho)
                                  + gm.normalized(lSho - thrx)
                                  + gm.normalized(thrx - rSho))
        tr[:3, 1] = gm.normalized(thrx - pelv)

        tr[:3, 2] = gm.normalized(np.cross(tr[:3, 0], tr[:3, 1]))
        tr[:3, 1] = gm.normalized(np.cross(tr[:3, 0], tr[:3, 2]))
        tr[:3, 3] = pelv  # - tr[:3, 1] * np.dot(tr[:3, 1], pelv)
        return tr

    def get_representative_frame(self):
        sclt_frames = self.get_frames()
        return sclt_frames[len(sclt_frames) // 2]

    def get_representative_time(self):
        frame_id = self.get_representative_frame()
        return self.get_time(frame_id)

    def get_transform_from_forward(self, dim=3, frame_id=-1, dtype=np.float32):
        """Creates a homogeneous transformation from thte position
        and the forward vector at a frame_id.
        If frame_id is negative, the middle frame_id is used.
        This is mostly used when aligning histograms to the path.
        :param frame_id: Which frame to use the forward and position of.
        Assumed to be the floor(middle frame), if negative.
        """
        if frame_id < 0:
            # get middle frame skeleton local coordinate frame in 2D
            # sclt_frames = self.get_frames()
            # frame_id = sclt_frames[len(sclt_frames) // 2]
            frame_id = self.get_representative_frame()
        fw = self.get_forward(frame_id, estimate_ok=False).astype(dtype)
        # homogeneous 2x3
        transform = np.identity(dim+1, dtype=dtype)
        if dim == 2:
            # x is forward
            transform[:2, 0] = gm.normalized(fw[[0, 2]])
            # z is orthogonal: orthogonal to (x, y) is (-y, x)
            transform[0, 1] = -transform[1, 0]
            transform[1, 1] = transform[0, 0]
            # translation is the pelvis at the time
            transform[:2, 2] = \
                self.get_joint_3d(
                    Joint.PELV, frame_id)[[0, 2]].astype(dtype)
            # rotate by 180.
            # transform[:2, :2] = np.dot(
            #     np.array([[-1, 0.], [0., -1]], dtype='f4'),
            #     transform[:2, :2])
        elif dim == 3:
            # x is forward
            transform[:3, 0] = gm.normalized(fw)
            # z is orthogonal: orthogonal to (x, y) is (-y, x)
            transform[0, 2] = -transform[2, 0]
            transform[2, 2] = transform[0, 0]
            # translation is the pelvis at the time
            transform[:3, 2] = \
                self.get_joint_3d(Joint.PELV, frame_id).astype(dtype)
            # rotate by 180.
            # transform[:3, :3] = np.dot(
            #     np.array([[-1, 0., 0.], [0., 1., 0.], [0., 0., -1]],
            #              dtype='f4'),
            #     transform[:3, :3])
        else:
            raise RuntimeError("Need dim==2 or dim==3, not %d" % dim)

        return transform

    def get_forward(self, frame_id, estimate_ok, k=2, debug=False,
                    estimate_force=False):
        """

        Args:
            estimate_ok (bool):
                Allow on-the-fly estimation from transform matrix.
            k: look this much both forward and backward
        Returns:
            forward (np.ndarray):
                Forward vector.
        """
        if not estimate_force and self._forwards is not None \
           and frame_id in self._forwards:
            return self._forwards[frame_id]
        elif estimate_force or estimate_ok:
            start = max(frame_id - k, min(self._frame_ids))
            end = min(frame_id + k, max(self._frame_ids))
            forwards = [
                Skeleton.get_forward_from_pose(self.get_pose(t))
                for t in range(start, end+1)
                if self.has_pose(t)
            ]
            forwards = [fw for fw in forwards
                        if not np.any(np.isnan(fw))]
            if len(forwards):
                return gm.normalized(np.mean(np.asarray(forwards).T, axis=1))
            else:
                raise ForwardEstimationError("No valid forwards")
        else:
            raise KeyError("Don't have interpolated forward for %d"
                               % frame_id)

    @staticmethod
    def get_forward_from_pose(pose):
        return Skeleton.get_forward_from_pose_v1(pose)

    def estimate_forwards(self, k=2):
        """Calculates forwards for each frame"""
        assert len(self._forwards) == 0, \
            "Already have some forwards...%s" % self._forwards
        for frame_id in self.get_frames():
            self.set_forward(
                frame_id,
                self.get_forward(frame_id=frame_id, estimate_ok=True, k=k,
                                 estimate_force=True),
                clone=True
            )

    @staticmethod
    def get_forward_from_pose_v2(pose):
        np.set_printoptions(linewidth=200, suppress=True)
        pca = PCA(n_components=2)
        # X = pose[[0, 2], :].T
        sel = [Joint.RHIP, Joint.LHIP, Joint.PELV, Joint.LSHO, Joint.RSHO, Joint.THRX]
        X = np.asarray([pose[0, sel], pose[2, sel]]).T
        S_pca_ = pca.fit(X).transform(X)
        # axes = S_pca_ / np.std(S_pca_, axis=0)
        # print("axes: %s" % axes)
        # axes /= axes.std()
        # print("axes normed: %s" % axes)
        print("comps: %s" % pca.components_)

        r_hip = pose[:, Joint.RHIP]
        lHip = pose[:, Joint.LHIP]
        rSho = pose[:, Joint.RSHO]
        lSho = pose[:, Joint.LSHO]
        pelv = pose[:, Joint.PELV]
        thrx = pose[:, Joint.THRX]
        lkne = pose[:, Joint.LKNE]
        rkne = pose[:, Joint.RKNE]
        lelb = pose[:, Joint.LELB]
        relb = pose[:, Joint.RELB]
        lank = pose[:, Joint.LANK]
        rank = pose[:, Joint.RANK]
        lwri = pose[:, Joint.LWRI]
        rwri = pose[:, Joint.RWRI]
        pairs = [ # first is important, that decides not to flip, if equal flips
            (lSho, rSho), (lHip, r_hip), (lHip, pelv), (pelv, r_hip)
            #(lSho, thrx), (thrx, rSho)
            # (lank, rank), (lwri, rwri)
        ]

        vectors = np.asarray([[pair[0][0] - pair[1][0],
                               pair[0][2] - pair[1][2]]
                              for pair in pairs])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        print("vectors: %s" % vectors)
        dots = np.dot(vectors, pca.components_)
        print("dots:  %s" % dots)
        sums = np.sum(abs(dots), axis=0)
        print(sums)
        if sums[0] < sums[1]:
            # rot = gm.rot_y(-np.pi / 2. if np.sum(dots[0, :]) < 0. else np.pi / 2)[:3, :3]
            # return np.dot(rot, [pca.components_[0, 0], 0., pca.components_[1, 0]])
            return np.asarray([pca.components_[0, 0], 0., pca.components_[1, 0]])
        else:
            # rot = gm.rot_y(-np.pi / 2. if np.sum(dots[1, :]) < 0. else np.pi / 2)[:3, :3]
            # return np.dot(rot, [pca.components_[0, 1], 0., pca.components_[1, 1]])
            return np.asarray([pca.components_[0, 1], 0., pca.components_[1, 1]])

        # print("initial vectors: %s" % vectors)
        # print("norms: %s" % np.linalg.norm(vectors, axis=1, keepdims=True))
        # vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        # print("normed vectors: %s" % vectors)
        # angles_orig = np.arctan2(vectors[:, 1], vectors[:, 0])
        # angles = angles_orig + np.pi
        # print("angles orig: %s" % [np.rad2deg(angle) for angle in angles])
        # angles *= np.float32(2.)
        # print("angles x 2: %s" % [np.rad2deg(angle) for angle in angles])
        # fractional, integral = np.modf(angles / np.pi / 2.)
        # angles -= integral * np.pi * 2.
        # # print("fractional: %s" % fractional)
        # # print("integral: %s" % integral)
        # print("angles x 2 mod 360.: %s" % [np.rad2deg(angle) for angle in angles])
        # vectors2 = np.vstack([np.cos(angles), np.sin(angles)]).T
        # print("vectors2: %s" % vectors2)
        # mean = np.mean(vectors2, axis=0)
        # mean /= np.linalg.norm(mean)
        # print("mean: %s" % mean)
        # angle_final = np.arctan2(mean[1], mean[0]) / 2.
        # print("final angle (atan2(mean): %s" % np.rad2deg(angle_final))
        # count = sum(1 for a in angles_orig if abs(angle_final - a) > np.pi / 2.)
        # print("flipcount: %s" % count)
        # if count > angles_orig.shape[0] // 2:
        #     angle_final += np.pi if angle_final < 0. else -np.pi
        # return np.dot(gm.rot_y(-angle_final + np.pi/2.)[:3, :3], [1., 0., 0.])

    @staticmethod
    def get_forward_from_pose_v1(pose):
        np.set_printoptions(linewidth=200, suppress=True)
        rhip = pose[:, Joint.RHIP]
        lhip = pose[:, Joint.LHIP]
        pelv = pose[:, Joint.PELV]
        thrx = pose[:, Joint.THRX]
        # lkne = pose[:, Joint.LKNE]
        # rkne = pose[:, Joint.RKNE]
        # lelb = pose[:, Joint.LELB]
        # relb = pose[:, Joint.RELB]
        # lank = pose[:, Joint.LANK]
        # rank = pose[:, Joint.RANK]
        # lwri = pose[:, Joint.LWRI]
        # rwri = pose[:, Joint.RWRI]

        rSho = pose[:, Joint.RSHO]
        lSho = pose[:, Joint.LSHO]
        pairs_all = [
            [
                (lSho, rSho), (lhip, rhip)  # (lhip, pelv), (pelv, rhip),
                # (lSho, thrx), (thrx, rSho)
                # (lank, rank), (lwri, rwri)
            ],
            [
                (lSho, thrx), (thrx, rSho), (lhip, pelv), (pelv, rhip)
            ]
        ]
        pairs = None
        for _pairs in pairs_all:
            pairs = _pairs
            vectors = [
                [pair[0][0] - pair[1][0], pair[0][2] - pair[1][2]]
                for pair in pairs
            ]
            vectors_norms = [
                v[0]*v[0] + v[1]*v[1]
                for v in vectors
            ]
            vectors = [
                v / np.sqrt(n)
                for v, n in zip(vectors, vectors_norms)
                if abs(n) > 0.
            ]
            # vectors2 = np.asarray([[pair[0][0] - pair[1][0],
            #                        pair[0][2] - pair[1][2]]
            #                       for pair in pairs])
            # vectors2 /= np.linalg.norm(vectors2, axis=1, keepdims=True)
            # lg.debug("vectors:\n%s" % vectors)
            # lg.debug("vectors2:\n%s" % vectors2)

            if len(vectors):
                vectors = np.array(vectors)
                break
            else:
                lg.error("pairs:\n%s\n, vectors:\n%s\n, norms:\n%s\n"
                         % (pairs, vectors, vectors_norms))
                pairs = None

            # # vectors_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # if not any(np.isnan(norm) or abs(norm) < 1e-3
            #            for norm in vectors_norms):
            #     vectors /= vectors_norms
            #     break
            # else:
            #     lg.error("pairs:\n%s\n, vectors:\n%s\n, norms:\n%s\n"
            #              % (pairs, vectors, vectors_norms))
            #     pairs = None
        # assert pairs is not None, \
        #     "Could not estimate forward: %s, %s" % (vectors, vectors_norms)
        if any(norm != norm for norm in vectors_norms):
            lg.debug("vector_norms: %s" % vectors_norms)
        if pairs is None:
            # raise ForwardEstimationError("Could not estimate valid forward")
            return np.array((np.nan, np.nan, np.nan))
        angles_orig = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles = angles_orig + np.pi
        angles *= np.float32(2.)
        fractional, integral = np.modf(angles / np.pi / 2.)
        angles -= integral * np.pi * 2.
        vectors2 = np.vstack([np.cos(angles), np.sin(angles)]).T
        mean = np.mean(vectors2, axis=0)
        mean /= np.linalg.norm(mean)
        angle_final = np.arctan2(mean[1], mean[0]) / 2.
        count = sum(1 for a in angles_orig if abs(angle_final - a) > np.pi / 2.)
        if count > angles_orig.shape[0] // 2:
            angle_final += np.pi if angle_final < 0. else -np.pi
        return np.dot(gm.rot_y(-angle_final + np.pi/2.)[:3, :3], [1., 0., 0.])

    def apply_transform(self, transform):
        """Transforms poses using homogeneous input transorm"""
        if transform.dtype != np.float32:
            lg.warning("Converting transform from %s" % transform.dtype)
            transform = transform.astype(np.float32)
        self._poses = gm.htransform(transform, self._poses)
        if self._forwards is not None and len(self._forwards):
            self._forwards = {
                frame_id_: np.dot(transform[:3, :3], forward_)
                for frame_id_, forward_ in self._forwards.items()
            }
            # for frame_id_, forward_ in self._forwards.items():
            #     self._forwards[frame_id_] = \
            #         np.dot(transform[:3, :3], forward_)
        self.set_changed(True)

    def get_angles(self, frame_id=None, estimate_ok=True):
        """Calculates angles between joints and their parents

        Args:
            frame_id (int):
                The frame we are interested in. None means all.
            estimate_ok (bool):
                Only return stored angles, don't estimate them.

        Returns:
            angles (np.ndarray):
                A row-vector of angles in radians.
        """
        frame_ids = self.get_frames() \
            if frame_id is None else [frame_id]
        if self._angles is not None and len(self._angles):
            angles = []
            for row_id, frame_id in enumerate(frame_ids):
                angles.append(self._angles[frame_id])
            angles = np.array(angles, dtype=np.float32)
            assert angles.shape[0] == len(frame_ids)
        elif estimate_ok:
            angles = np.zeros((len(frame_ids), Joint.get_num_joints()-4),
                              dtype=np.float32)
            joint_ids = {joint_id for joint_id in range(Joint.get_num_joints())
                         if joint_id != int(Joint.PELV)
                         and Joint(joint_id).get_parent() != int(Joint.PELV)}
            lin_frame_id = 0
            for frame_id in frame_ids:
                lin_angle_id = 0
                for joint_id in joint_ids:
                    parent = Joint(joint_id).get_parent()
                    j3d = self.get_joint_3d(joint_id, frame_id)
                    p3d = self.get_joint_3d(parent, frame_id)
                    pp3d = self.get_joint_3d(parent.get_parent(), frame_id)
                    v_lower = j3d - p3d
                    v_upper = pp3d - p3d
                    angles[lin_frame_id, lin_angle_id] = \
                        np.arctan2(np.linalg.norm(np.cross(v_lower, v_upper)),
                                   np.dot(v_lower, v_upper))
                    lin_angle_id += 1
                assert frame_ids[lin_frame_id] == frame_id, \
                    "Wrong: %d vs %d" % \
                    (frame_ids[lin_frame_id], frame_id)
                lin_frame_id += 1
        else:
            raise KeyError("Don't have angles for frame %d" % frame_id)

        return angles.astype(np.float32), frame_ids

    def get_min_y(self, tr_ground):
        """
        
        :param tr_ground: Ground transform
        :return: Scalar ground distance from origin 
        """
        assert tr_ground.shape == (4, 4), \
            "Wrong shape: %s" % tr_ground.shape.__repr__()
        up_axis = gm.normalized(tr_ground[:3, 1])

        # rot_ground = gm.get_rotation(tr_ground)
        # maxY = np.ones((3, 1), dtype=np.float32) * -sys.float_info.max
        # minY = np.ones((3, 1), dtype=np.float32) * sys.float_info.max
        # inv_tr = np.linalg.inv(rot_ground)
        max_d = -sys.float_info.max
        min_d = sys.float_info.max
        for frame_id in self.get_frames():
            pose = self.get_pose(frame_id)
            dots = np.matmul(up_axis.T, pose)
            assert dots.shape == (16,), \
                "Wrong shape: %s" % dots.shape.__repr__()
            max_d = max(max_d, np.max(dots))
            min_d = min(min_d, np.min(dots))
            # aa_pose = np.dot(inv_tr, pose)
            # print("pose[%d]:\n%s" % (frame_id, pose))
            # print("aa_pose[%d]:\n%s" % (frame_id, aa_pose))
            # maxY = np.maximum(np.max(aa_pose, 1), maxY)
            # minY = np.minimum(np.min(aa_pose, 1), minY)
        # print("AAmx: %s, AAmn: %s" % (maxY.T, minY.T))
        # maxY = np.dot(rot_ground, maxY)
        # minY = np.dot(rot_ground, minY)
        # print("mx: %s, mn: %s" % (maxY.T, minY.T))
        # print("rot_ground: %s" % rot_ground)
        # print("max_d: %f" % max_d)

        return max_d, min_d

    def set_changed(self, changed):
        self._changed = changed
        if self._changed:
            self._bone_obbs.clear()
            self._changed = False

    def get_bone(self, child_joint, frame_id):
        assert child_joint != Joint.PELV, \
            "PELV is root, bones are identified by the child joint"
        lin_id = self._frame_ids[frame_id]
        vec_child = self.get_joint_3d_by_lin_id(child_joint, lin_id)
        vec_parent = \
            self.get_joint_3d_by_lin_id(child_joint.get_parent(), lin_id)
        return vec_child - vec_parent, vec_parent

    def get_bone_obb(self, joint, frame_id):
        try:
            return self._bone_obbs[(frame_id, joint)]
        except KeyError:
            pass

        bone_vec, bone_start = self.get_bone(joint, frame_id)
        bone_obb = Obb.from_vector(bone_vec, scale=0.05)
        bone_obb.centroid += np.expand_dims(bone_start, 1)
        self._bone_obbs[(frame_id, joint)] = bone_obb
        return bone_obb

    def is_visible(self, frame_id, joint):
        """Check, if we know, that a joint was occluded"""
        assert frame_id in self._frame_ids, "No such frame: %d" % frame_id
        try:
            return self._visibility[frame_id][joint]
        except KeyError:
            return True

    def set_visible(self, frame_id, joint, visible):
        """Set visibility of a joint at time.

        Args:
             frame_id (int):
             joint (Joint):
             visible (bool):
        """
        assert frame_id in self._frame_ids, "No such frame: %d" % frame_id
        try:
            self._visibility[frame_id][joint] = int(bool(visible))
        except KeyError:
            self._visibility[frame_id] = {joint: int(bool(visible))}

    def set_confidence(self, frame_id, joint, confidence):
        """Set detection confidence of a joint at time.

        Args:
            frame_id (int):
            joint (Joint):
            confidence (float):
        """

        assert frame_id in self._frame_ids, "No such frame: %d" % frame_id
        try:
            self._confidence[frame_id][joint] = confidence
        except KeyError:
            self._confidence[frame_id] = {joint: confidence}

    @property
    def confidence(self):
        """Joint detection confidence for a frames and joints.

        Returns:
            confidence (Dict[int, Dict[int, float]]):
                {frame_id => {joint_id => confidence}}.
        """
        return self._confidence

    def get_confidence(self, frame_id, joint):
        """Returns the confidence of the joint at frame.

        Args:
            frame_id (int):
                Frame id.
            joint (Joint):
                Joint id.

        Returns:
            confidence (float):
                Joint confidence score.
        """

        return self._confidence[frame_id][joint]

    def get_max_confidence(self):
        return max(self._confidence[frame_id][joint]
                   for frame_id in self._confidence
                   for joint in self._confidence[frame_id])

    def has_confidence(self, frame_id, joint=None):
        """

        Args:
            frame_id (int):
            joint (Joint):
        Returns:
            has_confidence (bool):
        """
        ret = frame_id in self._confidence
        if joint is None:
            return ret
        else:
            return joint in self._confidence[frame_id]

    def get_confidence_matrix(self, frame_ids, dtype='f4'):
        """Get's a matrix that matches the dimensionality of _poses.

        Args:
            frame_ids (List[int]): The frame ids.
            dtype (Type): Output type ('b1' or 'f4').
        Returns:
            confidence (np.ndarray):
                The joint detection confidences.
        """
        visibility = \
            np.zeros(shape=(self._poses.shape[0], 1, self._poses.shape[2]),
                     dtype=dtype)
        cnt = 0
        # for frame_id in self._confidence:
        for frame_id in frame_ids:
            lin_id = self._frame_ids[frame_id]
            for joint_id in range(self._poses.shape[2]):
                try:
                    visibility[lin_id, 0, joint_id] = \
                        self._confidence[frame_id][joint_id]
                except KeyError:
                    assert joint_id in (Joint.NECK, Joint.PELV), \
                        'Expected missing confidence to be NECK or PELV, {}' \
                            .format(joint_id)
                    visibility[lin_id, 0, joint_id] = 0.
                cnt += 1
        assert cnt == visibility.size, "No: %s %s" % (cnt, visibility.size)
        return visibility.astype(dtype)

    def get_visibility_matrix(self, dtype='b1'):
        """Get's a matrix that matches the dimensionality of _poses"""
        visibility = \
            np.ones(shape=(self._poses.shape[0], 1, self._poses.shape[2]),
                    dtype=dtype)
        for frame_id in self._visibility:
            lin_id = self._frame_ids[frame_id]
            for joint_id in self._visibility[frame_id]:
                visibility[lin_id, 0, joint_id] = \
                    self._visibility[frame_id][joint_id]
        return visibility.astype(dtype)

    def has_visible(self, frame_id, joint_id=None):
        try:
            entry = self._visibility[frame_id]
        except KeyError:
            return False
        if joint_id is None:
            return True
        else:
            return joint_id in entry

    def is_confidence_normalized(self):
        """True, if confidence is already normalized to 0..1.

        Returns:
            is_confidence_normalized (bool):
                Is confidence already 0..1. Defaults to False due to LFD.
        """

        return hasattr(self, '_confidence_normalized') \
               and self._confidence_normalized

    def fill_with_closest(self, frame_start, frame_end):
        """Make sure there are no holes in the reconstruction"""
        frame_ids = self.get_frames()
        for frame_id in range(frame_start, frame_end+1):
            if frame_id not in frame_ids:
                closest_frame_id = min(frame_ids, key=lambda e: abs(e-frame_id))
                self.set_pose(frame_id, self.get_pose(closest_frame_id))
                frame_ids = self.get_frames()

    def to_mdict(self, name):
        """Converts to dictionary compatible with MATLAB"""
        mdict = {'poses': self._poses,
                 'frame_ids': self._frame_ids.items(),
                 'descriptors': None,
                 'name': [name],
                 'forward': [],
                 'frame_ids_float': self._times
                 }
        angles, frame_ids = self.get_angles()
        mdict['descriptors'] = np.hstack((
            np.asarray(frame_ids).reshape(-1, 1),
            angles))
        for frame_id in self.get_frames():
            mdict['forward'].append(
                np.hstack((frame_id,
                           self.get_forward(frame_id, estimate_ok=True, k=2))))
        return mdict

    def save_matlab(self, path_out, mdict=None, name='skeleton'):
        """Saves important info to MATLAB"""
        import scipy.io
        scipy.io.savemat(path_out,
                         self.to_mdict(name) if mdict is None else mdict)
        print("Saved to %s" % path_out)

    def compress_time(self):
        frames = self.get_frames()
        lin_id = frames[0]
        visibility = {}
        confidence = {}
        forwards = {}
        angles = {}
        frame_ids = {}
        for frame_id in frames:
            # visibility
            try:
                visibility[lin_id] = self._visibility[frame_id]
            except KeyError:
                pass

            # confidence
            try:
                confidence[lin_id] = self._confidence[frame_id]
            except KeyError:
                pass

            # forwards
            try:
                forwards[lin_id] = self._forwards[frame_id]
            except KeyError:
                pass

            # angles
            try:
                angles[lin_id] = self._angles[frame_id]
            except KeyError:
                pass

            # frame_id
            frame_ids[lin_id] = self._frame_ids[frame_id]

            lin_id += 1
        self._visibility = visibility
        self._confidence = confidence
        self._forwards = forwards
        self._angles = angles
        self._frame_ids = frame_ids
        self._frames_cache = None

        self.set_changed(True)

    def center_time(self):
        """Change frame_ids so that they match the center of the times.
        """
        if len(self._times):
            times = self._times
            mid_time_id = next(iter(times)) + len(times) // 2
            mid_frame_id = max(
                mid_time_id,
                int(round(times[mid_time_id])) \
                    if len(times) % 2 \
                    else int(round(times[mid_time_id] + times[mid_time_id+1]))
            )
            self.move_in_time(mid_frame_id - self.get_representative_frame())
        else:
            lg.error("No times")

    @staticmethod
    def resample(skeleton, frame_ids=None):
        """

        Args:
            skeleton (Skeleton):
                Class instance to resample in time densely.
            frame_ids (list): (List[int])
                Where to re-sample. If None, all frames between first and last
                will be assigned an interpolated 3D pose.
        """
        frames = skeleton.get_frames()
        frames_new = list(range(frames[0], frames[-1])) if frame_ids is None \
            else frame_ids

        resampled = np.transpose(
            np.array([
                [np.interp(x=frames_new, xp=frames,
                           fp=skeleton.poses[:, dim, joint_id])
                 for dim in range(3)]
                for joint_id in range(Joint.get_num_joints())],
                dtype=skeleton.poses.dtype
            ),
            axes=(2, 1, 0),
        )
        assert resampled.dtype == np.float32, \
            "Wrong type: %s" % resampled.dtype

        out = copy.deepcopy(skeleton)
        out._poses = resampled
        out._frame_ids = dict(zip(frames_new, range(len(frames_new))))
        out._frames_cache = None
        # visibility is default False for new frames, so that's ok
        assert not len(out._forwards), "TODO: interpolate forwards..."
        assert not len(out._angles), "Can't handle angles here"
        assert not len(out._bone_obbs), "Deprecated"
        assert not len(out._visibility), "TODO: shift visibility in time"
        assert not len(out._confidence), "TODO: shift confidence in time"
        if len(skeleton._times):
            times_new = np.interp(
               x=frames_new, xp=list(skeleton._times.keys()),
               fp=list(skeleton._times.values())
            )
            out._times = dict(zip(frames_new, times_new))
        assert resampled.shape[1] == 3, \
            "Wrong shape: %s" % repr(resampled.shape)
        assert resampled.shape[2] == Joint.get_num_joints(), \
            "Wrong shape: %s" % repr(resampled.shape)
        return out

    @staticmethod
    def get_resampled_centroids(start, end, old_frame_ids, poses):
        """Resamples only poses in 3D
        Args:

        """
        frames_new = list(range(start, end))

        centroids = np.mean(poses, axis=2)
        assert centroids.shape[0] == poses.shape[0], repr(centroids.shape)
        assert centroids.shape[1] == poses.shape[1]

        resampled = np.transpose(
          np.array([np.interp(x=frames_new, xp=old_frame_ids, fp=centroids[:, dim])
                    for dim in range(3)],
                   dtype=poses.dtype),
          axes=(1, 0),
        )
        assert resampled.shape[0] == (end - start), \
            (repr(poses.shape), repr(resampled.shape))
        assert resampled.shape[1] == poses.shape[1]
        return resampled

    def has_forwards(self):
        """Tells, if we have forwards for all frames.

        Returns:
            has_forwards (bool):
        """
        return len(self._forwards) == len(self._frame_ids)

    @lru_cache(maxsize=1)
    def get_rate(self):
        """Estimates the smallest time difference between two frame_ids."""
        return min(np.diff(list(self._times.values())))

    def guess_time_at(self, frame_id):
        """Estimates time for the given frame_id assuming constant rate."""

        if self.has_time(frame_id):
            return self.get_time(frame_id)
        else:
            frame_id_first, frame_id_last = self.get_frames_min_max()
            rate = self.get_rate()
            # towards beginning
            fi_ = frame_id - 1
            while not self.has_time(fi_) and fi_ >= frame_id_first:
                fi_ -= 1
            if fi_ >= frame_id_first:
                return self.get_time(fi_) + (frame_id - fi_) * rate

            # towards end
            fi_ = frame_id + 1
            while not self.has_time(fi_) and fi_ <= frame_id_last:
                fi_ += 1
            if fi_ <= frame_id_first:
                return self.get_time(fi_) + (frame_id - fi_) * rate
            else:
                raise RuntimeError("no frame backwards and forwards?")

    def compute_n_actors(self):
        """Estimates, how many actors there are stored in this skeleton."""
        if self._frames_mod is None:
            return 1
        max_frame_id = max(f for f in self._frame_ids)
        frame_id = self.mod_frame_id(max_frame_id, self._frames_mod)
        actor_id = 1
        while self.unmod_frame_id(frame_id, actor_id, self._frames_mod) < \
          max_frame_id:
            actor_id += 1
        return actor_id

    @property
    def n_actors(self):
        """

        Returns:
            n_actors (int):
                How many actors stored in skeleton.
        """
        return self._n_actors or 1

    @property
    def frames_mod(self):
        """The number of images in the video.

        Returns:
            frames_mod (int):
                number of images in the video.
        """

        return self._frames_mod

    @property
    def min_frame_id(self):
        """First frame_id in the video.

        Returns:
            min_frame_id (int):
                The first frame_id in the video.
        """

        return self._min_frame_id

    def get_actor_empty_frames(self):
        """Pairs of frame_ids where there is no information.

        At the transition times between actors, there might be frames where
        we already don't know anything about one actor but haven't seen the
        new actor yet. These spans are important to track to prevent
        interpolation and smoothing between actors.

        Returns:
            spans (tuple):
                ((last_frame_id_actor0, first_frame_id_actor1), ...)
                Note, frame ids returned exists, the gaps are between them.
        """
        return tuple(
          (
              self.get_actor_last_frame(actor_id=actor_id),
              self.get_actor_first_frame(actor_id=actor_id + 1)
              if actor_id + 1 < self.n_actors
              else self.get_frames()[-1] + 1)
          for actor_id in range(self.n_actors))
