from imapper.logic.joints import Joint


class JointDenis(object):
    _mapping = {
        0: Joint.PELV,
        1: Joint.RHIP,
        2: Joint.RKNE,
        3: Joint.RANK,
        4: Joint.LHIP,
        5: Joint.LKNE,
        6: Joint.LANK,
        # 7: Joint.STRN,
        8: Joint.THRX,
        9: Joint.NECK,
        10: Joint.HEAD,
        11: Joint.LSHO,
        12: Joint.LELB,
        13: Joint.LWRI,
        14: Joint.RSHO,
        15: Joint.RELB,
        16: Joint.RWRI
    }
    """Mapping from theirs to ours"""

    _mapping_2d = {
        0: Joint.HEAD,
        1: Joint.THRX,
        2: Joint.RSHO,
        3: Joint.RELB,
        4: Joint.RWRI,
        5: Joint.LSHO,
        6: Joint.LELB,
        7: Joint.LWRI,
        8: Joint.RHIP,
        9: Joint.RKNE,
        10: Joint.RANK,
        11: Joint.LHIP,
        12: Joint.LKNE,
        13: Joint.LANK
    }
    """Mapping from theirs to ours. NOTE: NECK and PELV is missing!"""

    revmap = [k for k, v in sorted(list(_mapping.items()), key=lambda e: e[1])]
    """Column mapping from theirs to ours, can be used to directly shuffle their
    pose array to ours via pose[:, revmap]"""
    revmap_2d = [k for k, v in sorted(list(_mapping_2d.items()), key=lambda e: e[1])]
    """Column mapping from theirs to ours, can be used to directly shuffle their
    pose array to ours via pose_2d[:, revmap_2d]"""

    @classmethod
    def to_ours(cls, joint):
        try:
            return cls._mapping[joint]
        except KeyError:
            raise RuntimeError("Unknown joint conversion for " + joint)

    @classmethod
    def to_ours_2d(cls, joint):
        try:
            return cls._mapping_2d[joint]
        except KeyError:
            raise RuntimeError("Unknown joint conversion for %s" % joint)

    @classmethod
    def pose_2d_to_ours(cls, pose_2d):
        assert pose_2d.shape == (14, 2), "Wrong shape: %s" % repr(pose_2d.shape)
        return [
            [pose_2d[10, 1], pose_2d[10, 0], 0.],  # RANK
            [pose_2d[9, 1], pose_2d[9, 0], 0.],    # RKNE
            [pose_2d[8, 1], pose_2d[8, 0], 0.],    # RHIP
            [pose_2d[11, 1], pose_2d[11, 0], 0.],  # LHIP
            [pose_2d[12, 1], pose_2d[12, 0], 0.],  # LKNE
            [pose_2d[13, 1], pose_2d[13, 0], 0.],  # LANK
            [(pose_2d[11, 1] + pose_2d[8, 1]) / 2.,
             (pose_2d[11, 0] + pose_2d[8, 0]) / 2., 0.],  # PELV
            [pose_2d[1, 1], pose_2d[1, 0], 0.],    # THRX
            [(pose_2d[1, 1] + pose_2d[0, 1]) / 2.,
             (pose_2d[1, 0] + pose_2d[0, 0]) / 2., 0.],   # NECK
            [pose_2d[0, 1], pose_2d[0, 0], 0.],    # HEAD
            [pose_2d[4, 1], pose_2d[4, 0], 0.],    # RWRI
            [pose_2d[3, 1], pose_2d[3, 0], 0.],    # RELB
            [pose_2d[2, 1], pose_2d[2, 0], 0.],    # RSHO
            [pose_2d[5, 1], pose_2d[5, 0], 0.],    # LSHO
            [pose_2d[6, 1], pose_2d[6, 0], 0.],    # LELB
            [pose_2d[7, 1], pose_2d[7, 0], 0.]     # LWRI
        ]

# print(sorted(list(_mapping.items()), key=lambda e: e[1]))
