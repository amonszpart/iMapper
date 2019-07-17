import sys
import os
import logging as _logging

try:
    from colorlog import ColoredFormatter
    lg = _logging.getLogger('stealth logger')
    lg.setLevel(_logging.DEBUG)

    _formatter = ColoredFormatter(
       "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s"
       "{new_line}%(pathname)s:%(lineno)d"
           .format(**{'new_line': "\t\t" if 'PYCHARM_HOSTED' not in os.environ else "\n"}),
       datefmt=None,
       reset=True,
       log_colors={
           'DEBUG':    'cyan',
           'INFO':     'green',
           'WARNING':  'yellow',
           'ERROR':    'red',
           'CRITICAL': 'red',
       }
    )
    ch = _logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(_logging.DEBUG)
    ch.setFormatter(_formatter)
    lg.addHandler(ch)
except ImportError:
    FORMAT = '[%(funcName)s] %(message)s\n%(pathname)s:%(lineno)d'
    _logging.basicConfig(level=_logging.DEBUG, format=FORMAT, stream=sys.stdout)
    lg = _logging

global logging
logging = lg


def split_path(path, prefix="\t"):
    """Splits a path to print into two lines:
        folder
        file_name
    """
    parts = os.path.split(path)
    return "%s/\n%s%s" % (parts[0], prefix, parts[1])
