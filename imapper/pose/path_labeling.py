import os
import sys
from imapper.pose.skeleton import JointDenis
from imapper.logic.joints import Joint
from imapper.logic.skeleton import Skeleton
from imapper.util.stealth_logging import lg
from imapper.config.conf import Conf
from imapper.pose.confidence import get_conf_thresholded
from imapper.visualization.plotting import plt
import cv2

try:
    import stealth.util.my_gurobi
    from gurobi import Model, GRB
except ImportError:
    print("Could not import Gurobi")
    pass

import copy
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.colors import Normalize as cmNormalize
from collections import namedtuple


import numpy as np


class ActorProblem(object):
    def __init__(self):
        self._vars = dict()
        self._pw = dict()
        self._constr = dict()

    def add_frame(self, frame_id, n_vars):
        for pose_id in range(n_vars):
            try:
                self._constr[frame_id].append(len(self._vars))
            except KeyError:
                self._constr[frame_id] = [len(self._vars)]

            self._vars[(frame_id, pose_id)] = len(self._vars)
        lg.debug("vars: %s" % self._vars)

    def get_n_vars(self):
        """How many variables overall"""
        return len(self._vars)

    def get_lin_id(self, frame_id, pose_id):
        return self._vars[(frame_id, pose_id)]

    def add_cost(self, frame_id, pose_id, frame_id1, pose_id1, cost):
        lin_id0 = self.get_lin_id(frame_id, pose_id)
        lin_id1 = self.get_lin_id(frame_id1, pose_id1)
        if lin_id0 > lin_id1:
            lin_id0, lin_id1 = lin_id1, lin_id0
            assert lin_id0 != lin_id1, "no: %s" % ((lin_id0, lin_id1))
        try:
            self._pw[(lin_id0, lin_id1)] += cost
        except KeyError:
            self._pw[(lin_id0, lin_id1)] = cost

    def get_frame_id_pose_id(self, lin_id):
        frame_id, pose_id = next(
            (frame_id, pose_id)
            for (frame_id, pose_id), lin_id_ in self._vars.items()
            if lin_id_ == lin_id)
        return frame_id, pose_id

class ActorProblem2(object):
    def __init__(self, n_actors, pose_not_present_unary_cost,
                 pose_not_present_pw_cost):
        self._n_actors = n_actors
        """How many actors to solve for."""
        self._pose_not_present_unary_cost = pose_not_present_unary_cost
        """Constant cost for "no detection" label."""
        self._pose_not_present_pw_cost = pose_not_present_pw_cost
        """Constant cost for switching between "no detection" label."""

        self._vars = dict()
        """Gurobi variables."""
        self._unary = dict()
        """Unary costs, e.g. sum of confidence for a 2d skeleton."""
        self._pw = dict()
        """Pairwise costs, e.g. distance between 2d skeletons."""
        self._constr_p = dict()
        """List of actor ids for each pose. (frame_id, actor_id) => [lin_id]"""
        self._constr_a = dict()
        """List of pose ids for each actor. (frame_id, pose_id) => [lin_id]"""
        self._constr_f = defaultdict(list)
        """List of visible actor ids for each frame. frame_id => [lin_id]"""

        self._solution = dict()
        """Output from solver. {var_id => value}"""

        self._min_counts = dict()

        self._inits = dict()
        self._max_pose_ids = dict()
        """frame_id => max_pose_id for frame"""

    def add_frame(self, frame_id, n_vars):
        self._max_pose_ids[frame_id] = n_vars
        for actor_id in range(self._n_actors):
            for pose_id in range(n_vars + 1):
                # next variable id
                lin_id = len(self._vars)

                # each pose can only get one actor label
                # including the "not present" label
                if pose_id != n_vars:
                    try:
                        self._constr_p[(frame_id, pose_id)].append(lin_id)
                    except KeyError:
                        self._constr_p[(frame_id, pose_id)] = [lin_id]

                # multiple actors can disappear, but each actor can only be
                # used once per frame
                # if actor_id != self._n_actors:
                try:
                    self._constr_a[(frame_id, actor_id)].append(lin_id)
                except KeyError:
                    self._constr_a[(frame_id, actor_id)] = [lin_id]

                self._constr_f[frame_id].append(lin_id)

                # add linear id for variable
                self._vars[(frame_id, pose_id, actor_id)] = lin_id
                self._inits[lin_id] = 0 if actor_id == self._n_actors \
                    else int(pose_id == actor_id)

    def get_n_vars(self):
        """How many variables overall"""
        return len(self._vars)

    def get_lin_id(self, frame_id, pose_id, actor_id):
        return self._vars[(frame_id, pose_id, actor_id)]

    def add_unary(self, frame_id, pose_id, cost):
        max_pose_id = self._max_pose_ids[frame_id]
        for actor_id in range(self._n_actors):
            lin_id = self.get_lin_id(frame_id, pose_id, actor_id)
            assert lin_id not in self._unary, "no"
            self._unary[lin_id] = cost

            lin_id_empty = self.get_lin_id(frame_id, max_pose_id, actor_id)
            self._unary[lin_id_empty] = self._pose_not_present_unary_cost

    def add_pw_cost(self, frame_id0, pose_id, frame_id1, pose_id1, cost):
        max_pose_id0 = self._max_pose_ids[frame_id0]
        max_pose_id1 = self._max_pose_ids[frame_id1]
        for actor_id in range(self._n_actors):
            # if actor_id0 == self._n_actors \
            #   and actor_id1 == self._n_actors:
            #     continue

            lin_id0 = self.get_lin_id(frame_id0, pose_id, actor_id)
            lin_id1 = self.get_lin_id(frame_id1, pose_id1, actor_id)
            # order
            if lin_id0 > lin_id1:
                lin_id0, lin_id1 = lin_id1, lin_id0

            key = (lin_id0, lin_id1)
            assert key not in self._pw, "no: %s" % repr(key)

            cost_ = cost
            # if actor_id1 == self._n_actors:
            #     cost_ = self._pose_not_present_pw_cost
            # elif actor_id0 != actor_id1:
            #     cost_ += self._pose_not_present_pw_cost / 4.

            self._pw[key] = cost_

            # to disappeared
            lin_id1_e = self.get_lin_id(frame_id1, max_pose_id1, actor_id)
            key = (lin_id0, lin_id1_e)
            if key in self._pw:
                assert abs(self._pw[key] - self._pose_not_present_pw_cost) \
                       < 1e-3, "Changing pw cost?"
            else:
                self._pw[key] = self._pose_not_present_pw_cost

            # from disappeared
            lin_id0_e = self.get_lin_id(frame_id0, max_pose_id0, actor_id)
            key = (lin_id0_e, lin_id1)
            if key in self._pw:
                assert abs(self._pw[key] - self._pose_not_present_pw_cost) \
                       < 1e-3, "Changing pw cost?"
            else:
                self._pw[key] = self._pose_not_present_pw_cost
            #
            # # from disappeared
            # lin_id0_e = self.get_lin_id(frame_id, pose_id, self._n_actors)
            # key = (lin_id0_e, lin_id1)
            # assert key not in self._pw, "no: %s" % repr(key)
            # # try:
            # #     self._pw[(lin_id0, lin_id1)] += self._pose_not_present_pw_cost
            # # except KeyError:
            # self._pw[key] = self._pose_not_present_pw_cost



    def get_frame_id_pose_id_actor_id(self, lin_id):
        frame_id, pose_id, actor_id = next(
          (frame_id, pose_id, actor_id)
          for (frame_id, pose_id, actor_id), lin_id_ in self._vars.items()
          if lin_id_ == lin_id)
        return frame_id, pose_id, actor_id

    def set_min_count(self, frame_id, min_count):
        self._min_counts[frame_id] = min_count

    def get_init_for_lin_id(self, lin_id):
        return self._inits[lin_id]


def identify_actors(data):
    m = Model('Stealth actors')

    problem = ActorProblem()

    objective = None
    prev_pose_in_2d = None
    prev_frame_id = None
    for frame_str in sorted(data):
        try:
            frame_id = int(frame_str.split('_')[1])
        except ValueError:
            print("skipping key %s" % frame_id)
            continue

        pose_in = np.array(data[frame_str][u'centered_3d'])
        pose_in_2d = np.array(data[frame_str][u'pose_2d'])
        visible = np.array(data[frame_str][u'visible'])
        assert pose_in_2d.ndim == 3, "no: %s" % repr(pose_in_2d.shape)

        problem.add_frame(frame_id, pose_in_2d.shape[0])

        if prev_pose_in_2d is not None:
            for prev_pose_id in range(prev_pose_in_2d.shape[0]):
                prev_pose = prev_pose_in_2d[prev_pose_id, :, :]
                for pose_id in range(pose_in_2d.shape[0]):
                    pose = pose_in_2d[pose_id, :, :]
                    dist = prev_pose - pose
                    lg.debug("dist: %s" % repr(dist.shape))
                    cost = np.sum(np.linalg.norm(dist, axis=1), axis=0)
                    lg.debug("cost: %s" % cost)
                    problem.add_cost(prev_frame_id, prev_pose_id, frame_id, pose_id, cost)

        prev_pose_in_2d = pose_in_2d
        prev_frame_id = frame_id

    gb_vars = m.addVars(problem.get_n_vars(), vtype=GRB.BINARY)

    for (lin_id0, lin_id1), cost in problem._pw.items():
        # lin_id0 = problem.get_lin_id(prev_frame_id, prev_pose_id)
        # lin_id1 = problem.get_lin_id(frame_id, pose_id)
        objective += gb_vars[lin_id0] * gb_vars[lin_id1] * cost

    for frame_id, lin_ids in problem._constr.items():
        constr = None
        for lin_id in lin_ids:
            if constr is None:
                constr = gb_vars[lin_id]
            else:
                constr += gb_vars[lin_id]
        m.addConstr(constr == 1)

    m.setObjective(objective, GRB.MINIMIZE)
    # m.solver.callSolver(m)
    m.optimize()

    pose_ids = dict()
    for lin_id, v in enumerate(m.getVars()):
        print(v.varName, v.x)
        if v.x > 0.5:
            frame_id, pose_id = problem.get_frame_id_pose_id(lin_id)
            assert frame_id not in pose_ids, "no"
            pose_ids[frame_id] = pose_id

    # if we have more, pick the first...
    # if len(pose_in.shape) > 2:
    #     pose_in = pose_in[0, :, :]
    #     pose_in_2d = pose_in_2d[0, :, :]
    #     visible = visible[0]

    return pose_ids


def show_images(images, data, thresh_log_conf=Conf.get().path.thresh_log_conf):
    _confs = []
    for frame_str in sorted(data):
        try:
            frame_id = int(frame_str.split('_')[1])
        except ValueError:
            print("skipping key %s" % frame_id)
            continue
        pose_in = np.asarray(data[frame_str][u'centered_3d'])
        pose_in_2d = np.asarray(data[frame_str][u'pose_2d'])
        visible = np.asarray(data[frame_str][u'visible'])
        vis_f = np.asarray(data[frame_str][u'visible_float'])

        # pose_id = pose_ids[frame_id]
        im = cv2.cvtColor(images[frame_id], cv2.COLOR_RGB2BGR)
        for i in range(pose_in.shape[0]):
            c = (.7, .2, .7, 1.)
            # if i == pose_id:
            #     c = (0., 1., 0., 1.)
            _confs.append(vis_f[i:i+1, :])
            color = tuple(int(c_ * 255) for c_ in c[:3])
            threshed = get_conf_thresholded(vis_f[i:i+1, :], dtype_np=np.float32,
                                            thresh_log_conf=thresh_log_conf)
            lg.debug("avg_vis: %s" % threshed)
            avg_vis = np.count_nonzero(threshed > 0.05, axis=1)
            # if avg_vis > 0.4:
            p2d_mean = np.mean(pose_in_2d[i, :, 1])
            cv2.putText(im, "%.2f" % (avg_vis / threshed.shape[1]),
                        (int(p2d_mean) - 20, 40), 1, 2, thickness=2,
                        color=(200, 200, 200))

            for j in range(pose_in_2d.shape[1]):
                p2d = pose_in_2d[i, j, :]
                conf = get_conf_thresholded(conf=vis_f[i, j],
                                            thresh_log_conf=thresh_log_conf,
                                            dtype_np=np.float32)
                if conf > 0.5:
                    cv2.circle(
                      im, (p2d[1], p2d[0]), radius=3, color=color, thickness=-1)
                    cv2.putText(im, ("%.2f" % conf)[1:],
                                (p2d[1], p2d[0]), 1, 1, color=color,
                                thickness=2)
                # if conf > 0.:
                #     cv2.putText(im, "%.2f" % avg_vis,
                #                 (p2d[1], p2d[0]), 1, 2, color=color)

            center = np.mean(pose_in_2d[i, :, :], axis=0).round().astype('i4').tolist()
            cv2.putText(im, "%d" % i, (center[1], center[0]), 1, 1, color)
        cv2.imshow("im", im)
        cv2.imwrite("/tmp/im_%04d.jpg" % frame_id, im)
        cv2.waitKey(100)
    # _confs = np.array(_confs).flatten()
    # _confs[_confs < 0.] = 0.
    # _confs = np.sort(_confs)
    # tr = 1. / (1. + np.exp(-5000. * _confs + 5))
    # print("here: %g" % np.exp(0.001))
    # plt.figure()
    # # plt.hist(_confs, bins=100)
    # sel = tr > 0.5
    # tr0 = tr[sel]
    # plt.scatter(_confs[sel], tr0, facecolor=None, edgecolor='g')
    # sel1 = tr < 0.5
    # plt.scatter(_confs[sel1], tr[sel1], facecolor=None, edgecolor='r')
    # plt.plot([0., 0.003], [0.5, 0.5], 'k')
    # plt.plot([0.001, 0.001], [0., 1.], 'k')
    # # plt.plot([0, len(_confs)], [np.log10(np.exp(-7.5)),
    # #                             np.log10(np.exp(-7.5))])
    # # plt.plot([0, len(_confs)], [np.exp(0.001) - 1., np.exp(0.001) - 1.])
    # # tr = 1. / (1 + np.log10(_confs))
    # # print("showing %s" % _confs)
    # plt.title('Transformed confidence: 1/(1 + exp(-5000x + 5))')
    # plt.xlim(-0.0001, 0.003)
    # plt.show()
    # plt.close()
    sys.exit(0)


def greedy_actors(data, n_actors):
    assert n_actors == 1, "not prepared for more actors"
    pose_ids = dict()
    for frame_str in sorted(data):
        try:
            frame_id = int(frame_str.split('_')[1])
        except ValueError:
            print("skipping key %s" % frame_id)
            continue

        pose_in = np.array(data[frame_str][u'centered_3d'])
        pose_in_2d = np.array(data[frame_str][u'pose_2d'])
        visible = np.array(data[frame_str][u'visible'])
        vis_f = np.array(data[frame_str][u'visible_float']) \
            if 'visible_float' in data[frame_str] \
            else None
        assert pose_in_2d.ndim == 3, \
            "no outer dim? %s" % repr(pose_in_2d.shape)
        if pose_in_2d.shape[0] == 1:
            pose_ids[frame_id] = [0]
            continue

        sum_confs = []
        for pose_id in range(pose_in_2d.shape[0]):
            conf = get_conf_thresholded(vis_f[pose_id, ...], None, np.float32)
            sum_confs.append(np.sum(conf))
        pose_ids[frame_id] = [int(np.argmax(sum_confs))]
    return pose_ids


class PosesWrapper(object):
    """Serves info about 2d poses to actor labeling optimization."""

    def __init__(self):
        pass


class SkeletonPosesWrapper(PosesWrapper):
    """Wraps skeleton files with multiple actors."""

    def __init__(self, skeleton):
        """Constructor

        Args:
            skeleton (Skeleton): skeleton
        """
        super(SkeletonPosesWrapper, self).__init__()
        assert skeleton.n_actors > 1, "Not multiactor?"
        self._skeleton = skeleton  # type: Skeleton
        self._pose_ids = {}  # type: dict
        """Remembers, which pose_id belongs to which frame and actor."""

    def get_frames(self):
        """

        Returns:
            frame_ids (set):
                Unique frame_ids of video, unmultiplexed.
        """
        return set([self._skeleton.mod_frame_id(frame_id=frame_id)
                    for frame_id in self._skeleton.get_frames()])

    ActorAndFrameIds = namedtuple('ActorAndFrameIds',
                                  ['actor_id', 'frame_id', 'frame_id2'])
    def get_poses_2d(self, frame_id):
        """2D pose for a given video frame_id.

        Args:
            frame_id (int):
        Returns:
            poses (np.ndarray): (n_actors, 3, 16)
        """
        skeleton = self._skeleton
        poses = []
        self._pose_ids[frame_id] = {}
        for actor_id in range(skeleton.n_actors):
            frame_id2 = skeleton.unmod_frame_id(
              frame_id=frame_id, actor_id=actor_id,
              frames_mod=skeleton._frames_mod
            )
            if skeleton.has_pose(frame_id2):
                pose_id = len(poses)
                self._pose_ids[frame_id][pose_id] = \
                    SkeletonPosesWrapper.ActorAndFrameIds(
                      actor_id=actor_id, frame_id=frame_id,
                      frame_id2=frame_id2)
                poses.append(skeleton.get_pose(frame_id=frame_id2))
        if len(poses):
            return np.transpose(np.array(poses), axes=(0, 2, 1))
        else:
            return np.zeros(shape=(0, 16, 2))

    def get_confidences(self, frame_id):
        """

        Args:
            frame_id (int): Frame id in question.

        Returns:
            confidences (np.ndarray): (N, 16)
                Array of confidences
        """
        N_JOINTS = Joint.get_num_joints()
        skeleton = self._skeleton
        confs = []  # type: List(List(float))
        for actor_id in range(skeleton.n_actors):
            frame_id2 = skeleton.unmod_frame_id(
              frame_id=frame_id, actor_id=actor_id,
              frames_mod=skeleton._frames_mod
            )
            if skeleton.has_pose(frame_id2):
                _confs = [
                    skeleton.get_confidence(frame_id=frame_id2, joint=j)
                    for j in range(N_JOINTS)]  # type: List(float)
                actor_and_frame_ids = self._pose_ids[frame_id][len(confs)]  # type: ActorAndFrameId
                assert actor_and_frame_ids.frame_id == frame_id \
                       and actor_and_frame_ids.frame_id2 == frame_id2 \
                       and actor_and_frame_ids.actor_id == actor_id
                confs.append(_confs)
            # else:
            #     lg.warning("Warning, no pose for %d %d"
            #                % (frame_id, frame_id2))

        return np.array(confs)

    def is_confidence_normalized(self):
        """

        Returns:
            normalized (bool):
        """
        assert self._skeleton.is_confidence_normalized(), \
            "Why not normalized? All LCR-Net skeletons should be..."
        return self._skeleton.is_confidence_normalized()

    def to_skeleton(self, pose_ids, skeleton3d):
        """Use 'pose_ids' to convert back to an ordered Skeleton.

        Args:
            pose_ids (Dict[int, Dict[int, int]]):
                {frame_id => {actor_id => pose_id}}.
            skeleton3d (Skeleton):
                Skeleton containing 3D poses.
        Returns:
            out2d (Skeleton):
                2D skeleton with sorted actors.
            out3d (Skeleton):
                3D skeleton with sorted actors.
        """

        skeleton = self._skeleton
        n_actors = skeleton.n_actors
        frames_mod = skeleton.frames_mod
        min_frame_id = skeleton.min_frame_id

        out2d = copy.deepcopy(skeleton)
        out2d.clear_poses()
        out3d = copy.deepcopy(skeleton3d)
        out3d.clear_poses()
        for frame_id in pose_ids:
            for actor_id in range(n_actors):
                # expanded frame_id
                frame_id2_dst = Skeleton.unmod_frame_id(frame_id=frame_id,
                                                        actor_id=actor_id,
                                                        frames_mod=frames_mod)
                #
                # checks
                #

                assert (actor_id != 0) ^ (frame_id2_dst == frame_id), "no"
                frame_id_mod = out3d.mod_frame_id(frame_id=frame_id2_dst)

                assert frame_id_mod == frame_id, \
                    "No: %d %d %d" % (frame_id, frame_id2_dst, frame_id_mod)
                actor_id2 = out3d.get_actor_id(frame_id2_dst)
                assert actor_id2 == actor_id, \
                    "No: %s %s" % (actor_id, actor_id2)

                #
                # Work
                #

                # which pose explains this actor in this frame
                pose_id = pose_ids[frame_id][actor_id]
                # check, if actor found
                if pose_id < 0:
                    continue

                actor_and_frame_ids = self._pose_ids[frame_id][pose_id]
                assert actor_and_frame_ids.frame_id == frame_id
                # assert actor_and_frame_ids.actor_id == actor_id
                frame_id2_src = actor_and_frame_ids.frame_id2

                # 3D pose
                pose3d = skeleton3d.get_pose(frame_id=frame_id2_src)  # type: np.ndarray
                time = skeleton3d.get_time(frame_id=frame_id2_src)
                out3d.set_pose(frame_id=frame_id2_dst, pose=pose3d, time=time)

                # 2D pose
                pose2d = skeleton.get_pose(frame_id=frame_id2_src)
                assert skeleton.get_time(frame_id=frame_id2_src) == time, \
                    "Time mismatch: %g %g" \
                    % (skeleton.get_time(frame_id=frame_id2_src), time)
                out2d.set_pose(frame_id=frame_id2_dst, pose=pose2d, time=time)

                # confidence
                for jid, conf in skeleton.confidence[frame_id2_src].items():
                    out3d.set_confidence(frame_id=frame_id2_dst,
                                         joint=jid, confidence=conf)
                    out3d.set_visible(frame_id=frame_id2_dst, joint=jid,
                                      visible=conf > 0.5)
                    out2d.set_confidence(frame_id=frame_id2_dst,
                                         joint=jid, confidence=conf)
                    out2d.set_visible(frame_id=frame_id2_dst, joint=jid,
                                      visible=conf > 0.5)

        # Testing
        assert out2d.is_confidence_normalized()
        assert out3d.is_confidence_normalized()

        return out2d, out3d


class DataPosesWrapper(PosesWrapper):
    """Wraps json files from LFD."""
    def __init__(self, data):
        super(DataPosesWrapper, self).__init__()
        self._data = data

    @staticmethod
    def _to_frame_str(frame_id):
        """Converts an integer frame_id to a string image name.

        Args:
            frame_id (int):
                Frame id in question.
        Returns:
            frame_str (str):
                Image identifier corresponding to frame id.
        """
        return "color_%05d" % frame_id

    def get_frames(self):
        return set([
            int(frame_str.split('_')[1]) for frame_str in sorted(self._data)
            if 'color' in frame_str])

    def get_poses_2d(self, frame_id):
        """

        Args:
            frame_id (int):
        Returns:
            poses (np.ndarray): (n_actors, 3, 16)
        """
        frame_str = DataPosesWrapper._to_frame_str(frame_id)  # type: str
        return np.array(self._data[frame_str][u'pose_2d'])

    def get_poses_3d(self, frame_id):
        """

        Args:
            frame_id (int):
        Returns:
            poses (np.ndarray): (n_actors, 3, 16)
        """
        frame_str = DataPosesWrapper._to_frame_str(frame_id)  # type: str
        return np.array(self._data[frame_str][u'centered_3d'])

    def get_confidences(self, frame_id):
        frame_str = DataPosesWrapper._to_frame_str(frame_id)  # type: str
        return np.array(self._data[frame_str][u'visible_float'])

    def get_visibilities(self, frame_id):
        frame_str = DataPosesWrapper._to_frame_str(frame_id)  # type: str
        return np.array(self._data[frame_str][u'visible'])

    def is_confidence_normalized(self):
        lg.warning("Assuming unnormalized confidence data.")
        return False


def more_actors_gurobi(data, n_actors, constraints,
                       first_run=False):
    """Multi-actor labeling using confidence weighted screen space distance.

    Args:
        first_run (bool):
            Short first run for vis only.
        n_actors (int):
            How many actors to label.
        constraints (Dict[str, Dict[int, int]]):
            {frame_str => {pose_id => actor_id}}.
        first_run (bool):
            Is this the very first run (limit runtime for only vis).
    Returns:
        pose_ids (Dict[int, Dict[int, int]]):
            {frame_id => {actor_id => pose_id}}
        problem (ActorProblem2):
            Labeling problem.
        data (PosesWrapper):
            Wrapped data for visualization.
    """

    # color_norm = cmNormalize(vmin=0, vmax=n_actors+1)
    # scalar_map = cm.ScalarMappable(norm=color_norm, cmap='gist_earth')
    # colors = [tuple(c * 255. for c in scalar_map.to_rgba(i+1))
    #           for i in range(n_actors)]
    # print(colors)
    # raise RuntimeError("")

    if isinstance(data, Skeleton):
        data = SkeletonPosesWrapper(skeleton=data)
    else:
        assert isinstance(data, dict), "%s" % type(data)
        data = DataPosesWrapper(data=data)
    is_conf_normalized = data.is_confidence_normalized()

    m = Model('Stealth actors')
    w_unary = 1.  # positive unary is a bonus
    pose_not_present_cost = w_unary * -1000  # negative unary is a penalty
    problem = ActorProblem2(n_actors=n_actors,
                            pose_not_present_unary_cost=pose_not_present_cost,
                            # positive pairwise is a penalty
                            pose_not_present_pw_cost=1000. * w_unary)

    objective = None
    prev_pose_in_2d = None
    prev_frame_id = None
    for frame_id in data.get_frames():
        # try:
        #     frame_id = int(frame_str.split('_')[1])
        # except ValueError:
        #     print("skipping key %s" % frame_id)
        #     continue
        frame_str = "color_%05d" % frame_id
        # if frame_id > 30:
        #     break
        # pose_in = np.array(data[frame_str][u'centered_3d'])
        # pose_in_2d = np.array(data[frame_str][u'pose_2d'])
        pose_in_2d = data.get_poses_2d(frame_id=frame_id)
        # visible = np.array(data[frame_str][u'visible'])
        # vis_f = np.array(data[frame_str][u'visible_float'])
        vis_f = data.get_confidences(frame_id=frame_id)
        assert pose_in_2d.ndim == 3, "no: %s" % repr(pose_in_2d.shape)

        problem.add_frame(frame_id,
                          n_vars=pose_in_2d.shape[0])

        # unary
        min_count = 0
        for pose_id in range(pose_in_2d.shape[0]):
            conf = vis_f[pose_id, ...]
            if not is_conf_normalized:
                conf = get_conf_thresholded(conf, thresh_log_conf=None,
                                            dtype_np=np.float32)
            cnt = np.sum(conf > 0.5)
            if cnt > conf.shape[0] // 2:
                min_count += 1
            unary = w_unary * np.sum(conf) / conf.shape[0]
            if frame_id == 251:
                print("here")
            # print("[%s] unary: %s" % (frame_id, unary))
            problem.add_unary(frame_id, pose_id, cost=unary)
        problem.set_min_count(frame_id, min_count)

        # pairwise
        if prev_pose_in_2d is not None:
            for prev_pose_id in range(prev_pose_in_2d.shape[0]):
                prev_pose = prev_pose_in_2d[prev_pose_id, :, :]
                for pose_id in range(pose_in_2d.shape[0]):
                    pose = pose_in_2d[pose_id, :, :]
                    dist = prev_pose - pose
                    dist = np.linalg.norm(dist, axis=1)
                    dist *= prev_vis_f[prev_pose_id, ...] * vis_f[pose_id, ...]
                    # lg.debug("dist: %s" % repr(dist.shape))
                    cost = np.sum(dist, axis=0)
                    # if cost > 200:
                    #     cost = 1e4

                    # cost /= 1500
                    # lg.debug("cost: %s" % cost)
                    problem.add_pw_cost(prev_frame_id, prev_pose_id,
                                        frame_id, pose_id, cost)

        prev_pose_in_2d = pose_in_2d
        prev_vis_f = vis_f
        prev_frame_id = frame_id

    gb_vars = m.addVars(problem.get_n_vars(), vtype=GRB.BINARY)
    # for lin_id in range(problem.get_n_vars()):
    #     gb_vars[lin_id].set(problem.get_init_for_lin_id(lin_id))

    # unary: we want to maximize confidence
    for lin_id, cost in problem._unary.items():
        objective -= gb_vars[lin_id] * cost

    # pairwise
    for (lin_id0, lin_id1), cost in problem._pw.items():
        objective += gb_vars[lin_id0] * gb_vars[lin_id1] * cost
    # print("NO PAIRWISE!!!")

    # a pose can only be labelled once per frame, either
    # actor0, or actor1, etc.
    for (frame_id, pose_id), lin_ids in problem._constr_p.items():
        constr = None
        for lin_id in lin_ids:
            if constr is None:
                constr = gb_vars[lin_id]
            else:
                constr += gb_vars[lin_id]
        # lg.debug("[%d] pose %d can only be %s" % (frame_id, pose_id, lin_ids))
        m.addConstr(constr <= 1)

    # an actor can only be used once per frame, either
    # pose0, or pose1, etc. Note: last actor can be used multiple times,
    # it's the "pose not present" label.
    for (frame_id, actor_id), lin_ids in problem._constr_a.items():
        constr = None
        for lin_id in lin_ids:
            if constr is None:
                constr = gb_vars[lin_id]
            else:
                constr += gb_vars[lin_id]
        # lg.debug("[%d] actor %d can only be %s"
        #          % (frame_id, actor_id, lin_ids))
        m.addConstr(constr == 1)

    # maximum number of poses chosen to be visible <= n_actors
    # for frame_id, lin_ids in problem._constr_f.items():
    #     constr = None
    #     for lin_id in lin_ids:
    #         if constr is None:
    #             constr = gb_vars[lin_id]
    #         else:
    #             constr += gb_vars[lin_id]
    #     m.addConstr(constr <= problem._n_actors)

    first_constrained = False  # type: bool
    min_frame_id = min(data.get_frames())  # type: int
    assert isinstance(min_frame_id, int)
    # anchor first pose as first actor
    if constraints and 'labels' in constraints:
        for frame_str, labels in constraints['labels'].items():
            frame_id = int(frame_str.split('_')[1])
            if isinstance(labels, list):
                assert len(labels) == n_actors, \
                    "frame: %d, %s %s" % (frame_id, len(labels), n_actors)
                # assert len(set(labels)) == len(labels), \
                #     "%s: %s" % (set(labels), labels)
                labels = {i: v for i, v in enumerate(labels)}
            for actor_id, pose_id in labels.items():
                pose_id = int(pose_id)
                if pose_id < 0:
                    pose_id = problem._max_pose_ids[frame_id]
                lin_id = problem.get_lin_id(frame_id, pose_id, actor_id)
                m.addConstr(gb_vars[lin_id] == 1)
                if not first_constrained and frame_id == min_frame_id \
                  and actor_id == 0:
                    first_constrained = True
    if not first_constrained:
        m.addConstr(gb_vars[0] == 1)
    # m.addConstr(gb_vars[36] == 1)
    # m.addConstr(gb_vars[40] == 1)

    m.setObjective(objective, GRB.MINIMIZE)
    m.Params.timeLimit = 300 if not first_run else 10
    # m.solver.callSolver(m)
    m.optimize()

    pose_ids = defaultdict(dict)
    prev_frame_id = None
    prev_lin_ids = {}
    curr_lin_ids = {}
    labelings = defaultdict(dict)
    for lin_id, v in enumerate(m.getVars()):
        frame_id, pose_id, actor_id = \
            problem.get_frame_id_pose_id_actor_id(lin_id)
        # print("[%d] %s: %s; pose %d is %sactor %s"
        #       % (frame_id, v.varName, v.x, pose_id,
        #          "not " if v.x < 0.5 else "", actor_id))
        problem._solution[lin_id] = v.x

        if prev_frame_id is not None:
            if prev_frame_id != frame_id:
                prev_lin_ids = copy.deepcopy(curr_lin_ids)
                curr_lin_ids.clear()

        # print("[#{f:d}][{l:d}] unary for p{p0:d}, a{a0:d} is {"
        #          "cost:f}{chosen:s}".format(
        #   f=frame_id, p0=pose_id, a0=actor_id,
        #   cost=problem._unary[lin_id], l=lin_id,
        #   chosen=" <-- chosen" if v.x > 0.5 else ""
        # ))

        if v.x > 0.5:
            curr_lin_ids[lin_id] = {"frame_id": frame_id,
                                    "pose_id": pose_id,
                                    "actor_id": actor_id}
            if pose_id == problem._max_pose_ids[frame_id]:
                pose_id = -1
            if frame_id in pose_ids and actor_id != n_actors:
                assert actor_id not in pose_ids[frame_id], "no"
            try:
                pose_ids[frame_id][actor_id] = pose_id
            except KeyError:
                pose_ids[frame_id] = {actor_id: pose_id}
            labelings[frame_id][pose_id] = actor_id

            # print("pw: %s" % problem._pw[lin_id])
            # for lin_id0, entries0 in prev_lin_ids.items():
            #     if (lin_id0, lin_id) in problem._pw:
            #         print("[#{f:d}] pw {l0:d}(p{p0:d},a{a0:d})"
            #               "->{l1:d}(p{p1:d},a{a1:d}) is {cost:f}".format(
            #           l0=lin_id0, l1=lin_id,
            #           cost=problem._pw[(lin_id0, lin_id)],
            #           a0=entries0['actor_id'], a1=actor_id,
            #           f=frame_id, p0=entries0['pose_id'], p1=pose_id
            #         ))


        prev_frame_id = frame_id

    # enforce constraints
    # if constraints and 'labels' in constraints:
    #     for frame_str, labels in constraints['labels'].items():
    #         frame_id = int(frame_str.split('_')[1])
    #         if isinstance(labels, list):
    #             labels = {v: i for i, v in enumerate(labels)}
    #         for pose_id, actor_id in labels.items():
    #             pose_ids[frame_id][actor_id] = int(pose_id)

    try:
        for frame_id in labelings:
            if frame_id % 5:
                continue
            print("\"color_%05d\": {%s},"
                  % (frame_id,
                     ", ".join(["\"%s\": %s" % (key, val)
                               for key, val in labelings[frame_id].items()])))
    except TypeError:
        pass


    # if we have more, pick the first...
    # if len(pose_in.shape) > 2:
    #     pose_in = pose_in[0, :, :]
    #     pose_in_2d = pose_in_2d[0, :, :]
    #     visible = visible[0]

    return pose_ids, problem, data

def show_multi(images, data, pose_ids, problem, p_dir,
               thresh_log_conf=Conf.get().path.thresh_log_conf,
               first_run=False, n_actors=1):
    """

    Args:
        images (Dict[int, np.ndarray]):
        data (SkeletonPosesWrapper):
        pose_ids (Dict[str, Dict[int, int]]):
        problem:
        p_dir (str):
        thresh_log_conf:
        first_run (bool):
            Will output labeling_orig if True allowing the inspection of
            pose_ids.
    """
    _confs = []
    # colors = {
    #     0: (.8, .1, .1, 1.),
    #     1: (.1, .8, .1, 1.),
    #     2: (.8, .8, .1, 1.),
    #     3: (.1, .8, .8, 1.),
    #     4: (.8, .1, .8, 1.),
    #     5: (.6, .4, .8, 1.),
    #     6: (.6, .4, .8, 1.)
    # }

    color_norm = cmNormalize(vmin=0, vmax=n_actors+1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='gist_earth')
    colors = [tuple(c for c in scalar_map.to_rgba(i+1))
              for i in range(n_actors)]

    p_labeling = os.path.join(p_dir, 'debug',
                              'labeling' if not first_run else 'labeling_orig')
    try:
        os.makedirs(p_labeling)
    except OSError:
        pass

    limits = (min(fid for fid in images), max(fid for fid in images)+1)
    scale = None
    for frame_id in range(limits[0], limits[1]):
        frame_str = "color_%05d" % frame_id
        # try:
        #     frame_id = int(frame_str.split('_')[1])
        # except ValueError:
        #     print("skipping key %s" % frame_id)
        #     continue
        # im = cv2.cvtColor(images[frame_id], cv2.COLOR_RGB2BGR)
        im = images[frame_id].copy()
        if im.shape[1] < 1900:
            if scale is None:
                scale = 1900 // im.shape[1] + 1
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
        elif scale is None:
            scale = 1.

        # for frame_id in data.get_frames():
        #     frame_str = "color_%05d" % frame_id
            # pose_in = np.asarray(data[frame_str][u'centered_3d'])
        pose_in_2d = data.get_poses_2d(frame_id=frame_id)

        # np.asarray(data[frame_str][u'pose_2d'])
        # visible = np.asarray(data[frame_str][u'visible'])
        # vis_f = np.asarray(data[frame_str][u'visible_float'])
        vis_f = data.get_confidences(frame_id=frame_id)

        # pose_id = pose_ids[frame_id]
        for pose_id in range(pose_in_2d.shape[0]):
            actor_id = next(
              (actor_id_
               for actor_id_, pose_id_ in pose_ids[frame_id].items()
               if pose_id == pose_id_),
              None)
            if actor_id is None:
                ccolor = (0.5, 0.5, 0.5, 1.)
            else:
                ccolor = colors[actor_id % len(colors)]
            _confs.append(vis_f[pose_id:pose_id+1, :])
            color = tuple(int(c_ * 255) for c_ in ccolor[:3])
            # threshed = get_conf_thresholded(vis_f[pose_id:pose_id+1, :],
            #                                 thresh_log_conf=thresh_log_conf,
            #                                 dtype_np=np.float32)
            # lg.debug("avg_vis: %s" % threshed)
            # avg_vis = np.count_nonzero(threshed > 0.05, axis=1)
            # if avg_vis > 0.4:
            p2d_mean = np.mean(pose_in_2d[pose_id, :, 1]) * scale
            # cv2.putText(im, "%.2f" % (avg_vis / threshed.shape[1]),
            #             (int(p2d_mean) - 20, 50), 1, 1, thickness=2,
            #             color=(200, 200, 200))
            if actor_id is None:
                actor_id = -1
            cv2.putText(im, "a%d" % actor_id,
                        (int(p2d_mean) - 20, 30), fontFace=1, fontScale=2,
                        thickness=2, color=tuple(_c * 0.2 for _c in color))

            for j in range(pose_in_2d.shape[1]):
                p2d = [int(round(c * scale))
                       for c in pose_in_2d[pose_id, j, :]]
                conf = get_conf_thresholded(conf=vis_f[pose_id, j],
                                            thresh_log_conf=thresh_log_conf,
                                            dtype_np=np.float32)
                if conf > 0.5:
                    cv2.circle(
                      im, (p2d[0], p2d[1]), radius=3, color=color,
                      thickness=-1)

                    # jid_ours = JointDenis.to_ours_2d(j)
                    jid_ours = j
                    cv2.putText(im, Joint(jid_ours).get_name(),
                                (p2d[0], p2d[1]-5), 1, 1, color=color,
                                thickness=1)

            center = (scale * np.mean(pose_in_2d[pose_id, :, :], axis=0)) \
                .round().astype('i4').tolist()
            # center = (scale * pose_in_2d[pose_id, 5, :])\
            #     .round().astype('i4').tolist()
            cv2.putText(im, "p%da%d" % (pose_id, actor_id),
                        (center[0], center[1]), 1, 2,
                        [c_ * 1.2 for c_ in color], thickness=2)


        # frame_id
        cv2.putText(im, "#%d" % frame_id, (20, 30), 1, 2, (255, 255, 255),
                    thickness=2)
        # cv2.imshow("im", im)
        p_im = os.path.join(p_labeling, "im_%04d.jpg" % frame_id)
        cv2.imwrite(p_im, im)
        # cv2.waitKey(20)
    # _confs = np.array(_confs).flatten()

