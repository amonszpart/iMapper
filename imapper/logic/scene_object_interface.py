import abc


class SceneObjectInterface(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, label):
        self.label = label
        """Object category, such as "couch", "chair" or "table"."""

    @abc.abstractmethod
    def get_centroid(self):
        raise NotImplementedError("get_centroid() not implemented")

    @abc.abstractmethod
    def get_transform(self):
        """Estimates the local->world transform of this object
            answering the questions "where is it?", and "what orientation?" 
        """
        raise NotImplementedError("get_transform() not implemented")

    @abc.abstractmethod
    def closer_to_scene_object_than(self, other, max_dist):
        raise NotImplementedError("closer_to_scene_object_than() not implemented")

    @abc.abstractmethod
    def apply_transform(self, transform):
        """ Transforms the object with a homog. transform
        :param transform: Homogeneous transformation matrix, 
                          will be pre-multiplied.
        :return: None
        """
        raise NotImplementedError("apply_transform() not implemented")
