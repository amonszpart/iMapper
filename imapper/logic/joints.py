from enum import IntEnum
import warnings


class Joint(IntEnum):
    RANK = 0
    RKNE = 1
    RHIP = 2
    LHIP = 3
    LKNE = 4
    LANK = 5
    PELV = 6
    THRX = 7
    NECK = 8
    HEAD = 9
    RWRI = 10
    RELB = 11
    RSHO = 12
    LSHO = 13
    LELB = 14
    LWRI = 15

    @classmethod
    def from_string(cls, string):
        return getattr(cls, string.upper(), None)

    @classmethod
    def get_ordered_range(cls):
        return [Joint.PELV, Joint.LHIP, Joint.LKNE, Joint.LANK, Joint.RHIP, Joint.RKNE, Joint.RANK, Joint.THRX,
                Joint.LSHO, Joint.LELB, Joint.LWRI, Joint.RSHO, Joint.RELB, Joint.RWRI, Joint.NECK, Joint.HEAD]

    @staticmethod
    def get_num_joints():
        """Equivalent to a 'LAST+1'.

        Returns:
            n_joints (int):
                Number of joints represented.
        """
        return int(Joint.LWRI) + 1

    def get_name(self):
        return self.__str__().split('.')[-1]

    @staticmethod
    def get_parent_of(joint):
        return Joint.__parents[joint]

    def get_parent(self):
        assert self in self.__parents, \
            "Can't find %s in %s" % (self, self.__parents)
        return self.__parents[self]
        # if self in {Joint.RANK, Joint.RKNE, Joint.RWRI, Joint.RELB}:
        #     return Joint(int(self) + 1)
        # elif self in {Joint.LANK, Joint.LKNE, Joint.LWRI,
        #               Joint.LELB, Joint.THRX, Joint.NECK, Joint.HEAD}:
        #     return Joint(int(self) - 1)
        # elif self in {Joint.RHIP, Joint.LHIP}:
        #     return Joint.PELV
        # elif self in {Joint.LSHO, Joint.RSHO}:
        #     return Joint.THRX
        # else:
        #     warnings.warn('Could not parse %d for parent' % self)
        # return None

    def get_bone_from_parent(self):
        """
        Unit length vector in direction from parent to this joint.
        :return: Tuple with length 1
        """
        # if self in [Joint.RANK, Joint.LANK, Joint.LKNE, Joint.RKNE, Joint.RELB, Joint.LELB, Joint.RWRI, Joint.LWRI]:
        #     return ((0., -1., 0.))
        if self in [Joint.LKNE, Joint.RKNE, Joint.RELB, Joint.LELB]:
            return (0., -1., 0.)
        elif self in [Joint.RANK, Joint.LANK, Joint.RWRI, Joint.LWRI]:
            return (0., 0., 1.)
        elif self is Joint.THRX:
            return (0., 1., 0.)
        elif self in [Joint.NECK, Joint.HEAD]:
            return (0., 0., 1.)
        elif self in [Joint.LSHO, Joint.LHIP]:
            return (-1., 0., 0.)
        elif self in [Joint.RSHO, Joint.RHIP]:
            return (1., 0., 0.)
        elif self in [Joint.PELV]:
            return (0., 0., 0.)
        else:
            raise ValueError('[get_bone_from_parent] Could not parse %d' % self)
            # warnings.warn('[get_bone_from_parent] Could not parse %d' % self)
        # return None


# def get_vector(j0, j1, skel):
    # print("skel[%d]: " % j0, skel[j0])
    # print("skel[%d]: " % j1, skel[j1])
    # return Vector(skel[j1]) - Vector(skel[j0])

# add class field afterwards
setattr(Joint, '_Joint__parents', {
    Joint.RANK: Joint.RKNE,
    Joint.RKNE: Joint.RHIP,
    Joint.RHIP: Joint.PELV,
    Joint.LANK: Joint.LKNE,
    Joint.LKNE: Joint.LHIP,
    Joint.LHIP: Joint.PELV,
    Joint.THRX: Joint.PELV,
    Joint.LSHO: Joint.THRX,
    Joint.RSHO: Joint.THRX,
    Joint.LELB: Joint.LSHO,
    Joint.RELB: Joint.RSHO,
    Joint.LWRI: Joint.LELB,
    Joint.RWRI: Joint.RELB,
    Joint.NECK: Joint.THRX,
    Joint.HEAD: Joint.NECK,
    Joint.PELV: None
})
