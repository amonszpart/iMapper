from imapper.logic.joints import Joint
import imapper.logic.geometry as gm
import numpy as np


def fw_normalized(fw, tr_ground=None):
    """Fetch position and normalized forward of skeleton at time"""
    if tr_ground is not None:
        fw = gm.normalized(gm.project_vec(fw, tr_ground))
    else:
        fw = gm.normalized(fw, tr_ground)
    return fw


def _get_angle(fw0, fw1, tr_ground):
    angle = gm.angle_3d(fw0, fw1)
    if tr_ground is not None and np.cross(fw0, fw1).dot(tr_ground[:3, 1]) < 0.:
        angle = -angle
    return angle


def get_angle(fws0, fws1, tr_ground=None):
    if not isinstance(fws0, list):
        fws0 = [fws0]
    if not isinstance(fws1, list):
        fws1 = [fws1]

    _fws0 = [fw_normalized(fw=fw0, tr_ground=tr_ground)
              for fw0 in fws0]
    _fws1 = [fw_normalized(fw=fw1, tr_ground=tr_ground)
             for fw1 in fws1]
    angles = [_get_angle(fw0, fw1, tr_ground)
              for fw0, fw1 in zip(_fws0, _fws1)]

    #
    # rotations
    #

    angle_vec = np.sum(
      np.array(
        [
            [
                np.float32(np.cos(a)),
                np.float32(np.sin(a))
            ]
            for a in angles
        ],
        dtype='f4'),
      axis=0
    ) / len(angles)
    angle = np.arctan2(angle_vec[1], angle_vec[0])

    return angle
