import os
import json

# try:
#     from types import SimpleNamespace as Namespace
# except ImportError:
#     Python 2.x fallback
    # from argparse import Namespace

#
# From argparse.py
#


class _AttributeHolder(object):
    """Abstract base class that provides __repr__.

    The __repr__ method returns a string in the format::
        ClassName(attr=name, attr=name, ...)
    The attributes are determined either by a class-level attribute,
    '_kwarg_names', or by inspecting the instance __dict__.
    """

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            arg_strings.append('%s=%r' % (name, value))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    def _get_kwargs(self):
        return sorted(self.__dict__.items())

    def _get_args(self):
        return []


class MyNamespace(_AttributeHolder):
    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            if not name.startswith('_comment'):
                setattr(self, name, kwargs[name])

    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, MyNamespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __ne__(self, other):
        if not isinstance(other, MyNamespace):
            return NotImplemented
        return not (self == other)

    def __contains__(self, key):
        return key in self.__dict__


class Conf(object):
    __instance = None
    __path_loaded = None
    __data = None

    @classmethod
    def get(cls, path=None):
        if cls.__instance is None:
            cls.__instance = Conf(path)
            cls.__data = cls.__instance.__data
            cls.__path_loaded = cls.__instance.__path_loaded
        return cls.__instance

    def __init__(self, path=None):
        if self.__data is None:
            if path is None:
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'conf.json')
            assert os.path.exists(path), "Can't find %s" % path
            with open(path, 'r') as fil:
                self.__data = json.load(
                    fil, object_hook=lambda d: MyNamespace(**d))
        else:
            assert path is None, "Want to init with a different path?"

    def __getattr__(self, item):
        # return self.__data[item]
        return self.__data.__dict__[item]

    def __repr__(self):
        return repr(self.__data)

if __name__ == '__main__':
    print(Conf.get())
    print(Conf.get().skeleton)