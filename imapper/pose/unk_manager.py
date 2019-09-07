import atexit
from enum import IntEnum

import numpy as np
import shapely.affinity as saffinity
import copy

from imapper.logic.categories import CATEGORIES, TRANSLATIONS_CATEGORIES
from imapper.logic.joints import Joint
from imapper.pose.confidence import get_confs
from imapper.scenelet_fit.create_dataset import get_poly
from imapper.util.stealth_logging import lg
from imapper.util.timer import Timer
from imapper.config.conf import Conf  # numeric constants
from imapper.pose.alignment import get_angle

_FTYPE_NP = np.float32
"""Data type for optimization, ensures the automatic type deduction
in TensorFlow is consistent.
"""


def scene_to_rects(scene_objects, resolution_mgrid, dtype):
    """

    Args:
        scene_objects (list): (List[Tuple[SceneObj, int, int]])
            List of scene objects, their transform ids and their category ids.
        resolution_mgrid (float):
            How close two sample points should be.
            Default: Conf.get().path.mgrid_res
        dtype (DataType):
            Floating point datatype.
    """
    assert resolution_mgrid > 0., "No: %s" % resolution_mgrid
    rects = []
    meshgrids = []
    for obj, idx_t, id_cat in scene_objects:
        poly = get_poly([part.obb for part in obj.parts.values()])

        ob_angle = obj.get_angle(positive_only=True)
        assert 0. <= ob_angle <= 2 * np.pi, "No: %g" % ob_angle
        tr = [np.cos(ob_angle), -np.sin(ob_angle),
              np.sin(ob_angle), np.cos(ob_angle),
              0, 0]
        poly2 = saffinity.affine_transform(
          poly,
          tr  # a, b, d, e, cx, cy
        )
        rect = np.array([
            [poly2.bounds[0], 0., poly2.bounds[1]],
            [poly2.bounds[2], 0., poly2.bounds[1]],
            [poly2.bounds[2], 0., poly2.bounds[3]],
            [poly2.bounds[0], 0., poly2.bounds[3]]
        ], dtype=dtype)
        # lg.debug("rect: %s" % rect)
        inv_transform = np.array(
          [[tr[0], 0., tr[2]],
           [0., 1., 0.],
           [tr[1], 0., tr[3]]])

        n_x = (poly2.bounds[2] - poly2.bounds[0]) / resolution_mgrid
        xs = np.linspace(poly2.bounds[0], poly2.bounds[2],
                         num=n_x, endpoint=True, dtype=dtype)
        n_y = (poly2.bounds[3] - poly2.bounds[1]) / resolution_mgrid
        ys = np.linspace(poly2.bounds[1], poly2.bounds[3], num=n_y,
                         endpoint=True, dtype=dtype)
        m_xs, m_ys = np.meshgrid(xs, ys)
        meshgrid = np.concatenate((
            m_xs.flatten()[:, None],
            np.zeros((m_xs.size, 1), dtype=dtype),
            m_ys.flatten()[:, None]), axis=1)
        meshgrids.append(np.matmul(inv_transform, meshgrid.T).T)
        rect = np.matmul(inv_transform, rect.T).T
        rects.append(rect)

        if False:
            from stealth.visualization.plotting import plt
            from descartes import PolygonPatch
            # rect = get_rectangle(poly, ob_angle)
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_artist(PolygonPatch(poly, facecolor='r', alpha=0.1))
            print(rect)
            plt.plot(rect[:, 0], rect[:, 2])
            plt.scatter(meshgrids[-1][:, 0], meshgrids[-1][:, 2])
            plt.xlim(-5, 5)
            plt.ylim(0, 10)
            plt.show()

    return rects, meshgrids


class PoseSource(IntEnum):
    """Flag for smoothness term, denotes the data source of 3D poses."""

    LFD = 0
    """Local 3D pose from LiftingFromTheDeep."""
    SCENELET_BEG = 1
    """First 3D pose of a scenelet."""
    SCENELET_END = 2
    """Last 3D pose of a scenelet."""
    SCENELET = 3
    """Collector of the above two, is invalid otherwise."""

    STATIC = 10
    """Poses that don't move (not optimized, no transforms)."""
    MOVING = 11
    """Poses that move (transforms optimized)."""

    def __str__(self):
        """String conversion for printing."""
        if self == PoseSource.SCENELET_BEG:
            return "SCENELET_BEG"
        elif self == PoseSource.SCENELET_END:
            return "SCENELET_END"
        elif self == PoseSource.SCENELET:
            return "SCENELET"
        elif self == PoseSource.LFD:
            return "LFD"
        elif self == PoseSource.STATIC:
            return "STATIC"
        elif self == PoseSource.MOVING:
            return "MOVING"
        else:
            raise RuntimeError("Unknown %d" % self)


class UnkManager(object):
    """Data manager that keeps track of the linear variables in the
    optimization problem and the semantic grouping of that data."""

    class Pid2Scene(object):
        """Tracks pose ids' association to input scenes, video frames and
        transforms."""

        def __init__(self, id_scene, id_scene_part, frame_id, transform_id):
            """Constructor."""

            self._id_scene = id_scene
            """Scene ID (which input scenelet did it come from) 
            Note: `scenes` are resorted based on starting frame_id inside 
            `match()`!.
            """

            self._id_scene_part = id_scene_part
            """Chunk ID in scene (e.g. scenelet id), is not used anymore,
            each scenelet is a \"scene\".
            """

            self.frame_id = frame_id
            """Query video frame_id that the pose id explains."""

            self.transform_id = transform_id
            """Transform ID, aka `idx_t`, indexes UnkManager._translations 
            and UnkManager._rotations. Not all poses have separate 
            transforms (e.g. poses in one scenelet move together)."""

        @property
        def id_scene(self):
            return self._id_scene

        @property
        def id_scene_part(self):
            return self._id_scene_part

        # @property
        # def frame_id(self):
        #     return self._frame_id

        # @property
        # def transform_id(self):
        #     return self._transform_id

    class SmoothEntry(object):
        """Represents a 3D pose added to the problem, used to determine
        smoothness connections.
        """

        def __init__(self, pose_id, source, moving):
            """Constructor."""

            assert source is not PoseSource.SCENELET, \
                "`SCENELET` not allowed, specify if _BEG or _END."

            self._pose_id = pose_id
            """Pose id in either _3d or _py_3d_static."""
            self._source = source
            """(PoseSource) is it a local pose or a scenelet start or end."""
            self._moving = moving
            """Is it optimized for (does it have a transform)."""

        @property
        def pose_id(self):
            return self._pose_id

        @property
        def source(self):
            return self._source

        @property
        def moving(self):
            return self._moving

        def __str__(self):
            return "SmoothEntry(pid={:d},source={:s},moving={:s}".format(
             self.pose_id, self.source.__str__(), self.moving.__str__()
            )

        def __repr__(self):
            return self.__str__()

    class Polys2Scene(object):
        """Description of a group of polygons in polys_3d."""

        def __init__(self, poly_id_start, n_polys, transform_id, cat_id,
                     part_label, object_id):
            """Constructor."""

            self._poly_id_start = poly_id_start
            """Index of first polygon of an object."""
            self._n_polys = n_polys
            """How many polygons belong to this object."""
            self._transform_id = transform_id
            """Transformation index, denotes which group/scenelet 
            the object belongs to."""
            self._cat_id = cat_id
            """Category id of object"""
            self._part_label = part_label
            """Category name of part of object"""
            self._object_id = object_id
            """Unique id of object this group of polygons belongs to."""

        @property
        def poly_id_start(self):
            return self._poly_id_start

        @property
        def n_polys(self):
            return self._n_polys

        @property
        def transform_id(self):
            return self._transform_id

        @property
        def cat_id(self):
            return self._cat_id

        @property
        def part_label(self):
            return self._part_label

        @property
        def object_id(self):
            """Note: This is not the original object id in the scenelet,
            but a unique one assigned in the problem."""
            return self._object_id

    def __init__(
      self, thresh_log_conf,
      cats_to_ignore,
      part_side_size_threshold=Conf.get().path.part_side_size_threshold,
      silent=False):
        """Constructor.
        Args:
          thresh_log_conf (float):
            Threshold to determine the visibility of a joint in the video.
            Is constant, and around -6..-7.
          part_side_size_threshold (float):
            Minimum threshold for an object part's side size to be added to
            the problem. Default: 0.05, i.e. `5 cm`.
        """

        self._2d = np.empty(shape=(0, 2, 16))
        """Shape: (N, 2, 16). 2D joint positions in the video."""
        self._translations = []  # translations
        """Shape: (M, 3). Translations optimized for. 
        Used to construct transforms.
        """
        self._cats_to_ignore = cats_to_ignore \
            if cats_to_ignore is not None else set()
        """Categories not participating in the optimization."""
        self._part_side_size_threshold = part_side_size_threshold
        """Don't add object parts that have a side smaller than this."""

        self._rotations = []  # rotations
        self._2d = []  # 2d detections
        self._3d = []  # 3d positions
        self._py_conf = []  # joint visibilities
        self._py_t_indices = []  # lin_id -> py_t
        self._max_poses_per_transform = 0
        """Maximum partition size for visibility estimation."""
        self._pids_2_scenes = {}  # lin_id in {py_2d: Pid2Scene}
        """Keeps track of pose ids, their times in the query video."""
        self._tids_pids = {}
        """Reverse cache for the above"""

        self._mx_conf = _FTYPE_NP(0.)
        self._explained_frame_ids = set()
        self._pids_to_smooth = {}
        """query_frame_id -> [(pid0, typ), (pid1, typ), ...]"""
        self._scene_objects = []
        self._pretransforms_scene = dict()
        self._obj_vertices = None
        self._obj_transform_indices = None
        # self._group_poly_ids = None
        self._pids_polyids = np.empty((0, 2), dtype=np.int32)
        """Moving point ids to moving poly ids for occlusion."""
        self._polys2scene = dict()
        """Keeps track of objects represented as groups of polygons. 
        {poly_id_start: Poly2Scene(), ...}"""

        # top-view polygons for intersection
        self._obj_2d_vertices = None
        self._obj_2d_transform_indices = None
        self._obj_2d_cat_ids_per_vertex = None
        self._obj_2d_cat_ids_per_poly = None
        self._obj_2d_angles_per_poly = None
        self._obj_2d_mgrid_vxs = None
        self._obj_2d_mgrid_transform_indices = None
        self._obj_2d_mgrid_cat_ids = None

        self._thresh_log_conf = thresh_log_conf
        """Deprecated"""

        # static poses
        # self._3d_static = []  # 3D positions
        # self._pids_to_smooth_static = {}
        # """query_frame_id -> [(pid0, typ), (pid1, typ), ...]"""

        self._tid_static = None
        """Last transformation ID is the static one."""
        self._tid_max = 0
        """Max number of transforms"""
        self._pid_static_first = None
        """Pose id of the first static pose."""
        self._static_pids_started = False

        self._finalized = False

        self._ignored_cats = set()
        if not silent:
            atexit.register(self.atexit)

    @staticmethod
    def compute_correspondence(query_start, query_end, frame_ids_scene,
                               fps, stretch, clip):
        assert stretch is False, "TODO"
        assert query_start <= query_end

        query_mid = query_start + (query_end - query_start) // 2
        scene_mid_lin = len(frame_ids_scene) // 2
        # scene_mid = frame_ids_scene[scene_mid_lin]
        out = []
        used_query = set()
        for id_pose, frame_id_scene in enumerate(frame_ids_scene):
            # frame_id_query = query_mid - (scene_mid - frame_id_scene) // fps
            frame_id_query = query_mid - int(round(
              (scene_mid_lin - id_pose) / fps))
            if clip and frame_id_query < query_start:
                continue
            if clip and frame_id_query > query_end:
                continue
            if frame_id_query not in used_query:
                used_query.add(frame_id_query)
                out.append((frame_id_query, frame_id_scene))
        return out

    def add_scenelet(self, corresp_frame_ids, query_2d_full, skeleton_scene,
                     query_3d_full, idx_t, id_scene, id_scene_part,
                     pose_source, tr_ground,
                     min_poses=5, dtype_np=np.float64):
        """Add a part of a 3D scene to the optimization.

        Arguments:
          corresp_frame_ids (List[int, int]):
            Contains (query_frame_id, scene_frame_id) pairs.
          query_2d_full (stealth.logic.skeleton.Skeleton):
            All 2D features to align to.
          skeleton_scene (stealth.logic.skeleton.Skeleton):
            3D poses to align to the 2d features.
          query_3d_full (stealth.logic.skeleton.Skeleton):
            Initialized world-space 3D poses (from LFD).
          idx_t (int):
            Index of transformation in _translations and _rotations.
          id_scene (int):
            Index of scene in scenes that.
          id_scene_part (int):
            Index of part of scene in scene. Deprecated.
          pose_source (PoseSource):
            Type of 3D pose: LFD or SCENELET.
          min_poses (int):
            How many correspondences are needed for a valid scenelet.
          dtype_np (data-type):
            Floating point type of the numpy data structures.
        """
        assert pose_source in (PoseSource.LFD, PoseSource.SCENELET), \
            "Unknown pose source: %s." % pose_source
        assert not self._static_pids_started, \
            "Once static pids have been appended, you can't add more " \
            "dynamic poses"
        if pose_source is PoseSource.LFD:
            assert query_3d_full is None, \
                "No query_3d_full to pre-align to, " \
                "if the input is from the same source."

        n_added = 0
        py_2d = []
        py_3d = []
        py_t_indices = []
        py_conf = []
        pids_to_smooth = {}
        explained_frame_ids = set()
        center_scene = np.array([0., 0., 0.], dtype=dtype_np)
        center_query = np.array([0., 0., 0., 0], dtype=dtype_np)
        pids_2_scenes = {}
        mx_conf = 0.
        pid_start = self.get_next_pid()
        frame_id_scene_first = None
        frame_id_scene_last = None
        fw_query = {} # input video rotations
        fw_scene = {} # scenelet forwards
        for id_pose, (frame_id_query, frame_id_scene) \
                in enumerate(corresp_frame_ids):
            if not query_2d_full.has_pose(frame_id_query):
                continue
            if not skeleton_scene.has_pose(frame_id_scene):
                continue
            pid = pid_start + len(py_2d)

            # keep track of 3D ends
            if frame_id_scene_first is None \
                    or frame_id_scene < frame_id_scene_first[0]:
                frame_id_scene_first = (frame_id_scene, frame_id_query, pid)
            if frame_id_scene_last is None \
                    or frame_id_scene > frame_id_scene_last[0]:
                frame_id_scene_last = (frame_id_scene, frame_id_query, pid)
            explained_frame_ids.add(frame_id_query)
            # assert pid not in lids_2_scenes
            # book-keeping
            assert pid not in pids_2_scenes, "pids are unique..."
            pids_2_scenes[pid] = UnkManager.Pid2Scene(
              id_scene=id_scene, frame_id=frame_id_query,
              id_scene_part=id_scene_part, transform_id=idx_t)

            # 2D points
            py_2d.append(query_2d_full.get_pose(frame_id_query)[:2, :])

            # 3D points
            pose = skeleton_scene.get_pose(frame_id_scene)
            py_3d.append(pose)
            center_scene += np.mean(pose, axis=1)
            if query_3d_full is not None \
                    and query_3d_full.has_pose(frame_id_query):
                center_query[:3] += np.mean(
                  query_3d_full.get_pose(frame_id_query), axis=1)
                center_query[3] += 1
                fw_query[id_pose] = query_3d_full.get_forward(
                  frame_id_query, estimate_ok=False)
                fw_scene[id_pose] = skeleton_scene.get_forward(
                  frame_id_scene, estimate_ok=False)

            # transformation index
            py_t_indices.append(idx_t)

            # confidence
            confs, mx_conf = get_confs(query_2d_full, frame_id_query,
                                       self._thresh_log_conf, mx_conf,
                                       dtype_np=dtype_np)
            py_conf.append(confs.tolist())

            n_added += 1
        if n_added < min_poses:
            # lg.warning("Not adding scenelet with %d valid correspondences"
            #            % n_added)
            return False

        #
        # Smoothness terms
        #

        entry_smooth = UnkManager.SmoothEntry(
          pose_id=frame_id_scene_first[2],
          source=(PoseSource.SCENELET_BEG
                  if pose_source == PoseSource.SCENELET else pose_source),
          moving=PoseSource.MOVING)
        try:
            pids_to_smooth[frame_id_scene_first[1]].append(entry_smooth)
        except KeyError:
            pids_to_smooth[frame_id_scene_first[1]] = [entry_smooth]
        # add end, if not the same as beginning
        if pose_source == PoseSource.SCENELET:
            assert frame_id_scene_first[1] != frame_id_scene_last[1], \
                "Same frame_id for start and end, very short scenelet...?"
            entry_smooth = UnkManager.SmoothEntry(
              pose_id=frame_id_scene_last[2],
              source=PoseSource.SCENELET_END,
              moving=PoseSource.MOVING)
            try:
                pids_to_smooth[frame_id_scene_last[1]].append(entry_smooth)
            except KeyError:
                pids_to_smooth[frame_id_scene_last[1]] = [entry_smooth]

        assert len(py_3d), "Need 3D poses to fit..."
        if pose_source == PoseSource.LFD:
            assert len(py_3d) == 1, "Not one pose?"
        center_scene /= dtype_np(len(py_3d))
        center_scene[1] = dtype_np(0.)
        # assert center_query[3] > 0, "No valid query poses: %s" % center_query

        py_3d = np.array(py_3d, dtype=dtype_np)
        py_3d -= center_scene[:, None]

        # init to original centroid
        t_init = center_scene.copy()
        # r_init = dtype_np(np.random.uniform(-np.pi/8., np.pi/8.))
        r_init = dtype_np(0.)

        # init to center of static poses
        if query_3d_full is not None:
            if center_query[3] <= 0:
                # find closest existing poses
                assert frame_id_scene_first[1] != frame_id_scene_last[1], \
                    "Same frame_id for start and end, very short scenelet...?"
                _mn, _mx = query_3d_full.get_frames_min_max()

                # seek to front
                frame_id_before = frame_id_scene_first[1] - 1
                while not query_3d_full.has_pose(frame_id_before) \
                        and frame_id_before >= _mn:
                    frame_id_before -= 1
                if frame_id_before < _mn \
                        or not query_3d_full.has_pose(frame_id_before):
                    lg.warning("Before limit exceeded")
                    return False

                # accumulate front
                center_query[:3] += np.mean(
                  query_3d_full.get_pose(frame_id_before), axis=1)
                center_query[3] += 1

                # seek to back
                frame_id_after = frame_id_scene_last[1] + 1
                while not query_3d_full.has_pose(frame_id_after) \
                        and frame_id_after <= _mx:
                    frame_id_after += 1
                if frame_id_after > _mx \
                        or not query_3d_full.has_pose(frame_id_after):
                    lg.warning("After limit exceeded")
                    return False

                # accumulate back
                center_query[:3] += np.mean(
                  query_3d_full.get_pose(frame_id_after), axis=1)
                center_query[3] += 1

            # estimate average translation of initial path
            center_query[:3] /= center_query[3]
            center_query[1] = dtype_np(0.)

            # add it as initialization
            t_init += center_query[:3]
            assert len(fw_query) > 1 and len(fw_scene) > 1, \
                "No forwards? %s %s" % (fw_query, fw_scene)
            min_id_pose = min(min(a, b) for a, b in zip(fw_scene, fw_query))
            max_id_pose = max(max(a, b) for a, b in zip(fw_scene, fw_query))
            assert min_id_pose < max_id_pose
            mid_pose = (max_id_pose - min_id_pose) // 2 + min_id_pose
            time_dists = [(abs(a - mid_pose), a) for a in fw_scene]
            time_dists = sorted(time_dists, key=lambda e: e[0])
            if len(time_dists) > 3:
                time_dists = time_dists[:3]
            fws_query = [fw_query[pose_id] for _, pose_id in time_dists]
            fws_scene = [fw_scene[pose_id] for _, pose_id in time_dists]
            angle = get_angle(fws_query, fws_scene, tr_ground=tr_ground)
            # print("angle between %s and %s is %s"
            #       % (fws_query, fws_scene, np.rad2deg(angle)))
            r_init = -angle


        success = self._add_entries(py_2d=py_2d,
                                    py_3d=py_3d,
                                    py_t_indices=py_t_indices,
                                    py_conf=py_conf,
                                    pids_to_smooth=pids_to_smooth,
                                    explained_frame_ids=explained_frame_ids,
                                    pids_2_scenes=pids_2_scenes,
                                    t_init=t_init,
                                    r_init=r_init,
                                    mx_conf=mx_conf)
        if success:
            self.add_pretransform_scene(id_scene, id_scene_part, idx_t,
                                        translation=-center_scene)
        return success

    def _add_entries(self, py_2d, py_3d, py_t_indices, py_conf,
                     pids_to_smooth, explained_frame_ids, pids_2_scenes,
                     t_init, r_init, mx_conf):
        """Add linearly stacked data points to the problem.

        Arguments:
            py_2d (np.ndarray): (N, 2, 16)
                2d feature points for each pose.
            py_3d (np.ndarray): (N, 3, 16)
                3d poses to align to video.
            py_t_indices (list): (N, 1)
                Transform indices, associations of poses to transforms.
            py_conf (list): (N, 16)
                Visibility in the video for each joint.
            pids_to_smooth (dict):
                Keeps track of types of poses at each video frame_id.
                {frame_id: [SmoothEntry], ...}
            explained_frame_ids (set):
                Which video frame_ids have already at least one 3D pose in
                the problem.
            pids_2_scenes (dict):
                Which pose ids (first index in py_2d, etc.) come from which
                scenes.
                {pid0: Pid2Smooth, pid1: ...}
            t_init (list): (N, 3)
                Translation initialization.
            r_init (list): (N,)
                Rotation initialization.
            mx_conf (float):
                Maximum visibility, for normalization in `finalize()`.
        Returns:
            True if successfully added.
        """
        assert not self._finalized, "Can't add more after finalization"

        # 2d positions
        assert len(py_2d) == len(py_3d), \
            "Assumed valid correspondences"
        self._2d.extend(py_2d)

        # 3d positions
        # assert len(py_3d) == py_3d.shape[0] and py_3d.shape[0] != 0
        assert len(py_3d) != 0
        self._3d.extend(py_3d)
        assert len(self._2d) == len(self._3d)

        # translations
        if t_init.ndim != 2 or t_init.shape != (1, 3):
            t_init = t_init.reshape((1, 3))
        self._translations.extend(t_init)
        # if not isinstance(r_init, np.ndarray) \
        #         or r_init.ndim != 2 or r_init.shape != (1, 1):
        #     r_init = np.array(r_init).reshape((1, 1))

        # rotations
        self._rotations.extend([r_init])
        assert len(self._translations) == len(self._rotations)

        # transform indices
        assert all(py_t_indices[0] == ti for ti in py_t_indices), \
            "Assumed a single transform id per call, needed for " \
            "max_point_count."
        # if not isinstance(py_t_indices, np.ndarray):
        #     py_t_indices = np.array(py_t_indices, dtype=np.int32)
        assert len(py_t_indices) == len(py_2d)
        assert max(py_t_indices) < len(self._translations), \
            "Invalid t ids"
        if len(self._py_t_indices):
            assert self._py_t_indices[-1] != py_t_indices[0], "Are you sure?"
        self._py_t_indices.extend(py_t_indices)
        assert len(self._py_t_indices) == len(self._2d)
        self._max_poses_per_transform = max(self._max_poses_per_transform,
                                            len(py_t_indices))

        # visibilities
        assert len(py_conf) == len(py_2d)
        self._py_conf.extend(py_conf)
        assert len(self._py_conf) == len(self._2d)

        # visibility normalization
        self._mx_conf = max(mx_conf, self._mx_conf)

        # explained video frame_ids
        self._explained_frame_ids.update(explained_frame_ids)

        # smoothness bookkeeping
        for q_frame_id, data in pids_to_smooth.items():
            assert isinstance(data, list)
            try:
                self._pids_to_smooth[q_frame_id].extend(data)
            except KeyError:
                self._pids_to_smooth[q_frame_id] = data

        # data source bookkeeping
        for k, v in pids_2_scenes.items():
            assert k not in self._pids_2_scenes
            self._pids_2_scenes[k] = v

        # report success
        return True

    def extract_scene_objects(self, scene, idx_t, no_legs=False):
        """
        Args:
            scene (Scenelet):
            idx_t (int):
                Transformation ID (per-scenelet).
            no_legs (bool):
                Save GPU memory by not adding legs.
        Returns:
            obj_tid_catids (List[Tuple]):

        """
        obj_tid_catids = []
        for oid, scene_object in scene.objects.items():
            # prune categories (e.g. books)
            if scene_object.label in self._cats_to_ignore:
                self._ignored_cats.add(scene_object.label)
                continue
            # convert labels
            try:
                cat = TRANSLATIONS_CATEGORIES[scene_object.label]
            except KeyError:
                cat = scene_object.label

            # find if used category (prune #2)
            try:
                cat_id = CATEGORIES[cat]
                if no_legs:
                    scene_object = copy.deepcopy(scene_object)
                    scene_object._parts = {
                        pid: part
                        for pid, part in scene_object._parts.items()
                        if 'legs' not in scene_object._parts}
                obj_tid_catids.append((scene_object, idx_t, cat_id))
            except KeyError:
                self._ignored_cats.add(cat)
        return obj_tid_catids

    def add_scene_objects(self, obj_tid_catids):
        """Add scene objects to problem.

        Args:
            obj_tid_catids (list):
                List assembled in extract_scene_objects.
                Contains tuples of (SceneObj, transform_index, cat_id).
        """
        self._scene_objects.extend(obj_tid_catids)
        # for oid, scene_object in scene.objects.items():
        #     if scene_object.label in ('book', 'wall', 'floor'):
        #         self._ignored_cats.add(scene_object.label)
        #         continue
        #     try:
        #         cat = TRANSLATIONS_CATEGORIES[scene_object.label]
        #     except KeyError:
        #         cat = scene_object.label
        #
        #     try:
        #         cat_id = CATEGORIES[cat]
        #         self._scene_objects.append((scene_object, idx_t, cat_id))
        #     except KeyError:
        #         self._ignored_cats.add(cat)

    def add_pose_static(self, pose, frame_id_query, skeleton_2d, pose_source,
                        id_scene, dtype_np):
        """Adds a 3D pose that's not optimized for and it's confidences for
        the occlusion loss.

        Args:
            pose (np.ndarray): (3, 16)
                3D skeleton that is not moved in the scene.
            frame_id_query (int):
                Frame_id in the video.
            skeleton_2d (stealth.logic.skeleton.Skeleton):
                2D feature points (detections).
            pose_source (PoseSource):
                Scenelet or LFD. Two scenelets don't interact in the
                smoothness.
            id_scene (int):
                Bookkeeping for export.
        """
        assert not self._finalized

        self._static_pids_started = True
        pid = self.get_next_pid()
        if self._pid_static_first is None:
            self._pid_static_first = pid
            self._set_tid_static()
        else:
            assert pid > self._pid_static_first
        smooth_entry = UnkManager.SmoothEntry(pose_id=pid, source=pose_source,
                                              moving=PoseSource.STATIC)
        try:
            self._pids_to_smooth[frame_id_query].append(smooth_entry)
        except KeyError:
            self._pids_to_smooth[frame_id_query] = [smooth_entry]

        assert pid not in self._pids_2_scenes, "pids are unique..."
        self._pids_2_scenes[pid] = UnkManager.Pid2Scene(
          id_scene=id_scene, frame_id=frame_id_query,
          id_scene_part=-1, transform_id=self.tid_static)

        # if pose.ndim == 2:
        #     pose = pose[None, :, :]
        # else:
        #     pose = pose.copy
        # self._3d_static = np.append(self._3d_static, pose, axis=0) \
        #     if len(self._3d_static) else pose
        self._3d.append(pose.copy())
        assert skeleton_2d.has_confidence(frame_id_query)
        confs = get_confs(skeleton_2d, frame_id_query,
                          self._thresh_log_conf, mx_conf=None,
                          dtype_np=dtype_np)
        self._py_conf.append(confs)

    def expand_to_smooth(self, joints_to_smooth, actor_spans=None):
        """
        Only adds a weighted smoothness term between two skeleton positions,
        if they both are not from scenelets, e.g. one of them is an individual
        pose.

        Args:
             joints_to_smooth (tuple):
                Joint ids that participate in the smoothing term.
                E.g. (RHIP, LHIP).
            actor_spans (tuple(tuple)):
                List of frame_ids inbetween which there should be no
                smoothness because it's two separate actors.
        Returns:
            A 2D array with the first two columns being integer indices
            to 3D poses, and the third being the weight
        """
        lg.warning("NOTE: we are using squared time in smoothness")
        # TODO: in the consistent case, lfd-lfd will be added even if it's
        #  spanned by a scenelet already...delete that connection by checking
        #  whether there is a scenelet entry
        out_indices = []         # [:, :3] and [:, 3:] are both moving
        out_indices_right_static = []  # [:, :3] is moving, [:, 3:] is static
        out_weights = []
        sorted_ = sorted(self._pids_to_smooth.items(), key=lambda e: e[0])
        assert any([e.source == PoseSource.LFD for e in sorted_[0][1]]), \
            "First frame needs to have a local pose...no?"
        prev_lfd = None
        for eid, (frame_id, pids) in enumerate(sorted_):
            # current local pose or previous one
            lfd = next(((e, frame_id)
                        for e in pids if e.source == PoseSource.LFD),
                       None)
            if lfd is None:
                lfd_before = prev_lfd
                lfd_after = next(((e, frame_id_)
                                  for frame_id_, pids_ in sorted_[eid+1:]
                                  for e in pids_
                                  if e.source == PoseSource.LFD))
            else:
                lfd_before, lfd_after = lfd, lfd

            assert lfd_before is not None, "No previous local pose?"
            assert lfd_before[1] <= frame_id
            assert lfd_after is not None, "No next local pose?"
            assert lfd_after[1] >= frame_id
            # lg.debug("\nlfd_before: %s\ncurr:%s\nnext:%s"
            #          % (lfd_before, frame_id, lfd_after))

            # sort pids so that scenelets come up first
            assert PoseSource.LFD < PoseSource.SCENELET_BEG \
                   and PoseSource.LFD < PoseSource.SCENELET_END
            pids_sorted = sorted(pids, key=lambda e: e.source,
                                 reverse=True)
            for smooth_entry in pids_sorted:
                if smooth_entry.source == PoseSource.LFD \
                  and smooth_entry.moving == PoseSource.STATIC:
                    continue

                # get lfd to connect to
                # for a scenelet, it's lfd_before, or after
                # for moving lfd, it's prev_lfd
                lfd_ = lfd_before \
                    if smooth_entry.source == PoseSource.SCENELET_BEG \
                    else (lfd_after
                          if smooth_entry.source == PoseSource.SCENELET_END
                          else prev_lfd)

                # skip first LFD1-LFD0, there is nothing to tie to
                # e.g. LFD0 will be None
                if lfd_ is None:
                    assert eid == 0 and smooth_entry.source == PoseSource.LFD
                    continue
                assert smooth_entry.moving != PoseSource.STATIC \
                    or lfd_.moving != PoseSource.STATIC, "Both static?"
                if actor_spans:
                    smaller, larger = (frame_id, lfd_[1]) \
                        if frame_id <= lfd_[1] \
                        else (lfd_[1], frame_id)
                    if smaller < larger:
                        do_ignore = next((True for span in actor_spans
                                          if span[0] < larger <= span[1]
                                          or span[0] <= smaller < span[1]), None)
                        if do_ignore:
                            lg.error("Ignoring %s %s\n%s" % (larger, smaller,
                                                             actor_spans))
                            continue
                delta_time = abs(frame_id - lfd_[1])
                weight = 1. / (delta_time * delta_time) \
                    if delta_time > 0. else 1.
                pid0 = smooth_entry.pose_id
                pid1 = lfd_[0].pose_id
                if smooth_entry.moving == PoseSource.STATIC:
                    pid0, pid1 = pid1, pid0
                elif pid0 > pid1 and smooth_entry.moving == PoseSource.STATIC:
                    assert False, "this should never happen..."
                    assert smooth_entry.moving == lfd_[0].moving \
                           and lfd_[0].moving == PoseSource.MOVING
                    pid0, pid1 = pid1, pid0

                # assert pid0 != pid1
                for joint_id in joints_to_smooth:
                    indices = [[pid0, 0, joint_id], [pid0, 1, joint_id],
                               [pid0, 2, joint_id], [pid1, 0, joint_id],
                               [pid1, 1, joint_id], [pid1, 2, joint_id]]
                    # if smooth_entry.moving == PoseSource.MOVING \
                    #    and smooth_entry.moving == lfd_[0].moving:
                    out_indices.append(indices)
                    # else:
                    #     out_indices_right_static.append(indices)
                    out_weights.append(weight)
            prev_lfd = lfd_before
        return out_indices, out_indices_right_static, out_weights

    def _create_objects_3d(self, do_interact_other_groups,
                           w_static_occlusion, dtype_np):
        assert not self._finalized, "Call once"

        n_polys = 6
        l_pids_2_scenes = self._pids_2_scenes

        moving_pids_cache = {}

        py_obj_vxs = []
        py_obj_t_indices = []
        py_group_poly_ids = []
        py_pids_polyids = []  # moving point ids and moving polygons
        # next_part_id_in_group = {}  # poly_id inside a scenelet, not used
        poly_id = 0
        for oid, (obj, transform_id, id_cat) in enumerate(self._scene_objects):
            pretransform = self.get_pretransform(transform_id)

            # occlusion
            try:
                moving_pids = moving_pids_cache[transform_id]
            except KeyError:
                if do_interact_other_groups:
                    if self._tid_static is not None: # we hav
                        tids_to_match = list(range(0, self.tid_static+1)) \
                            if w_static_occlusion \
                            else list(range(0, self.tid_static))
                    else:
                        # get number of transforms
                        tids_to_match = list(
                          range(0, self.get_next_tid(no_check=True)))

                    moving_pids_cache[transform_id] = [
                        (pid, l_pids_2_scenes[pid].frame_id)
                        for tid in tids_to_match
                        for pid in self.get_pids_for(tid)]
                else:
                    tids_to_match = set((transform_id, self.tid_static)) \
                        if w_static_occlusion and self._tid_static is not None\
                        else [transform_id]
                    moving_pids_cache[transform_id] = [
                        (pid, l_pids_2_scenes[pid].frame_id)
                        for tid in tids_to_match
                        for pid in self.get_pids_for(tid)]

                moving_pids = moving_pids_cache[transform_id]

            # if transform_id not in next_part_id_in_group:
            #     next_part_id_in_group[transform_id] = 0

            for part in obj.parts.values():
                rect = part.obb.rectangles_3d().reshape((-1, 3))
                rect = np.matmul(pretransform[:3, :3], rect.T).T \
                    + pretransform[:3, 3]
                py_obj_vxs.extend(rect)
                py_obj_t_indices.extend(
                  [transform_id for _ in range(rect.shape[0])])
                assert len(py_obj_vxs) == len(py_obj_t_indices)

                # grouping to scenelets
                # next_part_id = next_part_id_in_group[transform_id]
                # to_add = [[transform_id, next_part_id + rect_id]
                #           for rect_id in range(n_polys)]
                # poly_id_start = len(py_group_poly_ids)
                # py_group_poly_ids.extend(to_add)
                # next_part_id_in_group[transform_id] = to_add[-1][1] + 1

                # book keeping for export scenelet
                self._polys2scene[poly_id] = UnkManager.Polys2Scene(
                  poly_id_start=poly_id, n_polys=n_polys,
                  transform_id=transform_id, cat_id=id_cat,
                  part_label=part.label, object_id=oid,
                )

                # filter small parts from occlusion term for efficiency
                if obj.label != 'table':
                    if any(scale < self._part_side_size_threshold
                           for scale in part.obb.scales):
                        poly_id += n_polys
                        continue

                # occlusion
                # gather which points this polygon can occlude
                py_pids_polyids.extend([[pid, poly_id + _poly_id, frame_id]
                                        for pid, frame_id in moving_pids
                                        for _poly_id in range(n_polys)])

                poly_id += n_polys

        self._obj_vertices = np.array(py_obj_vxs, dtype=dtype_np)
        self._obj_transform_indices = np.array(py_obj_t_indices,
                                               dtype=np.int32)
        # not used
        # self._group_poly_ids = np.array(py_group_poly_ids,
        #                                 dtype=np.int32)
        lg.warning("Don't sort, flatten pre-structured array...")
        with Timer('sorting by pids'):
            py_pids_polyids = sorted(py_pids_polyids, key=lambda e: e[0])
        self._pids_polyids = np.array(py_pids_polyids, dtype=np.int32)
        assert self._pids_polyids.ndim == 2 \
            and self._pids_polyids.shape[1] == 3

    def _create_objects_2d(self, resolution_mgrid, dtype_np):
        """Creates top-view rectangles.
        Args:
            resolution_mgrid (float):
                Meshgrid sample point distance. Should be
                Conf.get().path.mgrid_res.
        """
        assert not self._finalized, "Call once"
        lg.info("[UnkManager] Starting _create_objects_2d...")

        scene_objects = self._scene_objects
        rects, mgrids = scene_to_rects(scene_objects,
                                       resolution_mgrid=resolution_mgrid,
                                       dtype=dtype_np)
        py_obj_vxs = []
        py_mgrid_vxs = []
        py_indices = []
        py_mgrid_indices = []
        py_cat_ids_polys = []
        py_cat_ids_vxs = []
        py_cat_ids_mgrids = []
        py_angles = []
        for (obj, idx_t, id_cat), rect, mgrid in zip(scene_objects,
                                                     rects, mgrids):
            pretransform = self.get_pretransform(idx_t)
            _rect = np.matmul(pretransform[:3, :3], rect.T).T
            _rect += pretransform[:3, 3]
            py_obj_vxs.extend(_rect)
            py_indices.extend([idx_t for _ in range(_rect.shape[0])])
            py_cat_ids_vxs.extend([id_cat for _ in range(_rect.shape[0])])
            # for vx in rect:
            #     py_obj_vxs.append(vx - py_t[idx_t, :])
            #     py_indices.append(idx_t)
            #     py_cat_ids_vxs.append(id_cat)
            py_cat_ids_polys.append(id_cat)
            angle = obj.get_angle(positive_only=True)
            py_angles.append(angle)
            assert len(py_obj_vxs) == len(py_indices)
            assert len(py_cat_ids_vxs) == len(py_obj_vxs)
            assert len(py_angles) == len(py_cat_ids_polys)

            _mgrid = np.matmul(pretransform[:3, :3], mgrid.T).T
            _mgrid += pretransform[:3, 3]
            py_mgrid_vxs.extend(_mgrid.tolist())
            # l_ = (mgrid - py_t[idx_t, :]).tolist()
            # py_mgrid_vxs.extend(l_)
            py_mgrid_indices.extend([idx_t for _ in range(_mgrid.shape[0])])
            py_cat_ids_mgrids.extend([id_cat for _ in range(_mgrid.shape[0])])
            assert len(py_mgrid_vxs) == len(py_mgrid_indices)
            assert len(py_mgrid_vxs) == len(py_cat_ids_mgrids)
        # py_obj_vxs = np.array(py_obj_vxs, dtype=dtype_np)
        # py_mgrid_vxs = np.array(py_mgrid_vxs, dtype=dtype_np)
        # py_indices = np.array(py_indices, dtype=np.int32)
        # py_mgrid_indices = np.array(py_mgrid_indices, dtype=np.int32)
        # py_cat_ids_polys = np.array(py_cat_ids_polys, dtype=np.int32)
        # py_cat_ids_vxs = np.array(py_cat_ids_vxs, dtype=np.int32)
        # py_cat_ids_mgrids = np.array(py_cat_ids_mgrids, dtype=np.int32)
        # py_angles = np.array(py_angles, dtype=dtype_np)

        self._obj_2d_vertices = \
            np.array(py_obj_vxs, dtype=dtype_np)
        self._obj_2d_transform_indices = \
            np.array(py_indices, dtype=np.int32)
        self._obj_2d_cat_ids_per_vertex = \
            np.array(py_cat_ids_vxs, dtype=np.int32)
        self._obj_2d_cat_ids_per_poly = \
            np.array(py_cat_ids_polys, dtype=np.int32)
        self._obj_2d_angles_per_poly = \
            np.array(py_angles, dtype=dtype_np)

        # mgrids for object-object
        self._obj_2d_mgrid_vxs = \
            np.array(py_mgrid_vxs, dtype=dtype_np)
        self._obj_2d_mgrid_transform_indices = \
            np.array(py_mgrid_indices, dtype=np.int32)
        self._obj_2d_mgrid_cat_ids = \
            np.array(py_cat_ids_mgrids, dtype=np.int32)

        lg.info("[UnkManager] Finished _create_objects_2d...")

    def _set_tid_static(self):
        assert self._tid_static is None
        self._tid_static = self.get_next_tid(no_check=True)
        self._tid_max = max(self._tid_static, self._tid_max)

    def finalize(self, scale_2d, intrinsics, with_intersection,
                 do_interact_other_groups, w_static_occlusion,
                 resolution_mgrid, dtype_np):
        """
        Args:
            scale_2d:
            intrinsics:
            with_intersection (bool):
                Do we need the 2d rectangles?
            w_static_occlusion (bool):
                Should we occlude already placed path?
            resolution_mgrid (float):
                Meshgrid sample point distance. Should be
                Conf.get().path.mgrid_res.
            dtype_np:
        Returns
        """
        # translations
        self._translations = np.array(self._translations,
                                      dtype=dtype_np).reshape((-1, 3))
        assert self._translations.shape[1] == 3 \
            and self._translations.ndim == 2
        self._translations.flags.writeable = False

        # rotations
        self._rotations = np.array(self._rotations,
                                   dtype=dtype_np).reshape((-1, 1))
        assert self._rotations.shape[0] == self._translations.shape[0] \
            and self._rotations.shape[1] == 1 and self._rotations.ndim == 2
        self._rotations.flags.writeable = False

        # confidences
        self._py_conf = np.array(self._py_conf, dtype=dtype_np)
        self._py_conf /= dtype_np(self._mx_conf)
        self._py_conf.flags.writeable = False

        # 2d poses
        # scale from Denis' scale to current image size
        self._2d = np.array(self._2d, dtype=dtype_np)
        self._2d *= scale_2d

        # move to normalized camera coordinates
        self._2d -= intrinsics[:2, 2:3]
        self._2d[:, 0, :] /= intrinsics[0, 0]
        self._2d[:, 1, :] /= intrinsics[1, 1]

        self._py_t_indices = np.array(self._py_t_indices, dtype=np.int32)
        assert self._py_t_indices.shape[0] == self._2d.shape[0]
        self._tid_max = max(self._tid_max, np.max(self._py_t_indices))


        self._3d = np.array(self._3d, dtype=dtype_np)
        assert self._3d.ndim == 3 and self._3d.shape[1] == 3 \
            and self._3d.shape[2] == 16, "%s" % repr(self._3d.shape)

        if self._tid_static is None and self._static_pids_started:
            self._set_tid_static()
            assert all(p.transform_id <= self._tid_static
                       for p in self._pids_2_scenes.values())
        # obb-s for occlusion and scenelet export
        lg.info("[UnkManager] Starting _create_objects_3d...")
        with Timer('create_objects_3d'):
            self._create_objects_3d(do_interact_other_groups,
                                    w_static_occlusion=w_static_occlusion,
                                    dtype_np=dtype_np)
        lg.info("[UnkManager] Finished _create_objects_3d...")

        # top-view rectangles for intersection
        if with_intersection:
            with Timer('create_objects_2d'):
                self._create_objects_2d(resolution_mgrid=resolution_mgrid,
                                        dtype_np=dtype_np)

        self._finalized = True  # some inter-queries depend on this


    #
    # Getters
    #

    @property
    def get_n_transforms(self):
        """Number of transform ids, including static"""
        return self._tid_max + 1

    def get_next_pid(self):
        return len(self._3d)

    # def get_next_pid_static(self):
    #     if not isinstance(self._3d_static, list):
    #         assert len(self._3d_static) == self._3d_static.shape[0]
    #
    #     return len(self._3d_static)

    def get_next_tid(self, no_check=False):
        if not no_check:
            assert not self._static_pids_started, \
                "If we have started adding static poses, no new transform ids " \
                "should be issued."
        return len(self._translations)

    @property
    def tid_static(self):
        assert self._tid_static is not None, "Finalize first"
        return self._tid_static

    def has_static(self):
        return self._tid_static is not None

    @property
    def pid_static_first(self):
        """Pose id of first static pose."""
        return self._pid_static_first

    def get_pretransform(self, transform_id):
        return self._pretransforms_scene[transform_id]

    @property
    def translations(self):
        assert self._finalized
        return self._translations

    @property
    def rotations(self):
        assert self._finalized
        return self._rotations

    @property
    def poses_2d(self):
        assert self._finalized
        return self._2d

    @property
    def poses_3d(self):
        assert self._finalized
        return self._3d

    @property
    def poses_3d_static(self):
        assert self._finalized
        return self._3d_static

    @property
    def confidence(self):
        assert self._finalized
        return self._py_conf

    @property
    def transform_indices(self):
        assert self._finalized
        return self._py_t_indices

    @property
    def pids_2_scenes(self):
        assert self._finalized
        return self._pids_2_scenes

    def get_pids_for(self, transform_id):
        try:
            return self._tids_pids[transform_id]
        except KeyError:
            assert not len(self._tids_pids), "Can't have any values here."
            self._tids_pids = {}
            for pid, pid2scene in self._pids_2_scenes.items():
                tid = pid2scene.transform_id
                try:
                    self._tids_pids[tid].append(pid)
                except KeyError:
                    self._tids_pids[tid] = [pid]

            return self._tids_pids[transform_id]

    def get_pids_interacting_with(self, transform_id):
        assert False
        # pids = self.get_pids_for(transform_id) +

    @property
    def object_vertices(self):
        return self._obj_vertices

    @property
    def object_transform_indices(self):
        return self._obj_transform_indices

    @property
    def explained_frame_ids(self):
        return self._explained_frame_ids

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def max_poses_per_transform(self):
        return self._max_poses_per_transform

    # @property
    # def group_poly_ids(self):
    #     return self._group_poly_ids

    @property
    def max_polys_per_transform(self):
        return np.max(self._group_poly_ids[:, 1])

    @property
    def polys2scene(self):
        return self._polys2scene

    @property
    def pids_polyids(self):
        assert self._finalized
        return self._pids_polyids

    @property
    def obj_2d_vertices(self):
        return self._obj_2d_vertices

    @property
    def obj_2d_transform_indices(self):
        return self._obj_2d_transform_indices

    @property
    def obj_2d_cat_ids_per_vertex(self):
        return self._obj_2d_cat_ids_per_vertex

    @property
    def obj_2d_cat_ids_per_poly(self):
        return self._obj_2d_cat_ids_per_poly

    @property
    def obj_2d_angles_per_poly(self):
        return self._obj_2d_angles_per_poly

    @property
    def obj_2d_mgrid_vxs(self):
        return self._obj_2d_mgrid_vxs

    @property
    def obj_2d_mgrid_transform_indices(self):
        return self._obj_2d_mgrid_transform_indices

    @property
    def obj_2d_mgrid_cat_ids(self):
        return self._obj_2d_mgrid_cat_ids

    #
    # Setters
    #

    def add_pretransform_scene(self, id_scene, id_scene_part, transform_id,
                               translation, transform=None):
        key = transform_id
        assert key not in self._pretransforms_scene, "duplicate pretransforms"
        if transform is None:
            transform = np.array([[1., 0., 0., translation[0]],
                                  [0., 1., 0., translation[1]],
                                  [0., 0., 1., translation[2]],
                                  [0., 0., 0., 1.]],
                                 dtype=translation.dtype)
        else:
            assert translation is None, \
                "Can't have both translation and transform"
        self._pretransforms_scene[key] = transform

    #
    # Util
    #

    def atexit(self):
        lg.warning("Ignored the following categories: %s" % self._ignored_cats)

    # def add_moving_pose(self, pose, query_2d_full, transform_id,
    #                     frame_id_query, dtype_np):
    #     """Adds a moving local pose with given transform_id.
    #
    #     Args:
    #         pose (np.ndarray):
    #           Shape: (3, 16). 3D pose to fit to the video.
    #         frame_id_query (int):
    #           The frame_id in the query video that this pose explains.
    #         query_2d_full (stealth.logic.Skeleton):
    #           All 2D keypoints from the video in xy.
    #         transform_id (int):
    #           Linear id of transformation in self._translations,
    #           self._rotations and transforms.
    #         dtype_np (np.type):
    #           Floating point data type. Default: np.float64.
    #     """
    #     assert pose.ndim == 2 and pose.shape[1] == Joint.get_num_joints(), \
    #         "One pose expected: %s" % pose
    #
    #     mx_conf = 0.
    #     pids_to_smooth = {}
    #
    #     # center to rotate pose around
    #     center_scene = np.mean(pose, axis=1)
    #     assert center_scene.size == 3, "No: %s" % center_scene
    #     center_scene[1] = 0.
    #
    #     # pose_id, unique id of the 3D cloud in self._py_3d
    #     pid = self.get_next_pid()
    #     pids_2_scenes = {}
    #     assert pid not in pids_2_scenes
    #     pids_2_scenes[pid] = UnkManager.Pid2Scene(
    #       id_scene=-1, frame_id=frame_id_query, id_scene_part=-1,
    #       transform_id=transform_id
    #     )
    #
    #     # 2D points
    #     py_2d = [query_2d_full.get_pose(frame_id_query)[:2, :]]
    #
    #     # 3D points
    #     py_3d = [pose]
    #
    #     # transformation index
    #     py_transform_indices = [transform_id]
    #
    #     # confidence
    #     confs, mx_conf = get_confs(query_2d_full, frame_id_query,
    #                                self._thresh_log_conf, mx_conf,
    #                                dtype_np=dtype_np)
    #     py_conf = [confs.tolist()]
    #
    #     entry_smooth = UnkManager.SmoothEntry(pose_id=pid,
    #                                           source=PoseSource.LFD,
    #                                           moving=PoseSource.Moving)
    #     try:
    #         pids_to_smooth[frame_id_query].append(entry_smooth)
    #     except KeyError:
    #         pids_to_smooth[frame_id_query] = [entry_smooth]
    #
    #     py_3d = np.array(py_3d, dtype=dtype_np)
    #     py_3d -= center_scene[:, None]
    #
    #     t_init = center_scene
    #
    #     return self._add_entries(py_2d=np.array(py_2d), py_3d=py_3d,
    #                              py_t_indices=py_transform_indices,
    #                              py_conf=py_conf,
    #                              pids_to_smooth=pids_to_smooth,
    #                              explained_frame_ids=set(),
    #                              pids_2_scenes=pids_2_scenes,
    #                              t_init=t_init, r_init=0.,
    #                              mx_conf=mx_conf)
