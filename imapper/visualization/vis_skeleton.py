import numpy as np

from imapper.logic.joints import Joint


class VisSkeleton:
    _colors_sizes = {
        Joint.LANK: ((.8, .2, .2), .05),
        Joint.LKNE: ((.8, .2, .2), .05),
        Joint.LHIP: ((.8, .2, .2), .075),
        Joint.RANK: ((.2, .2, .8), .05),
        Joint.RKNE: ((.2, .2, .8), .05),
        Joint.RHIP: ((.2, .2, .8), .075),
        Joint.LSHO: ((.8, .5, .2), .075),
        Joint.LELB: ((.8, .5, .2), .05),
        Joint.LWRI: ((.8, .5, .2), .05),
        Joint.RSHO: ((.2, .5, .8), .075),
        Joint.RELB: ((.2, .5, .8), .05),
        Joint.RWRI: ((.2, .5, .8), .05),
        Joint.HEAD: ((0.57254902, 0.396078431, 0.733333333), .15),
        Joint.PELV: ((1., 0.921568627, 0.803921569), .05),
        Joint.THRX: ((1., 0.921568627, 0.803921569), .075),
        Joint.NECK: ((.7, .8, .4), .1)
    }

    @classmethod
    def vis_skeleton(cls, vis, pose, prefix, flip=None, valid=None,
                     color_add=None, forward=None):
        pose = pose.copy()
        color_mult = 1.
        if valid is not None and valid != 1:
            color_mult = 0.5

        # forward = Skeleton.get_forward_from_pose(pose)
        if forward is not None:
            pose[:, Joint.NECK] = \
                pose[:, Joint.THRX] \
                + (pose[:, Joint.HEAD] - pose[:, Joint.THRX]) / 2. \
                + forward / 2.

        for joint in range(pose.shape[1]):
            color = np.asarray(cls._colors_sizes[joint][0]) * color_mult
            if color_add is not None:
                color += color_add
            vis.add_sphere(
                pose[:, joint],
                radius=cls._colors_sizes[joint][1],
                color=color,
                name="%s_%02d" % (prefix, joint))
