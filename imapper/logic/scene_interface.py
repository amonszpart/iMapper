import abc
from collections import Counter


class SceneInterface(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        """Name of the scene from its txt filename"""
        self._objects = dict()
        """Objects organized by their object ids"""

    def save(self, path):
        raise NotImplementedError("Need implementation")

    def get_labels(self, ignore=set()):
        """Returns an occurrence list of object categories"""
        l = [o.label for o in self.objects.values() if o.label not in ignore]
        return dict(Counter(l).items())

    @property
    def objects(self):
        return self._objects
