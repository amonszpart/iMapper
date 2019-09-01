import sys
if sys.version_info[0] >= 3:
    import pickle as pickle
    def pickle_load(f):
        return pickle.load(f, encoding='latin1')
else:
    import cPickle as pickle
    def pickle_load(f):
        return pickle.load(f)
import hashlib

# def load_pickle(path):
#     if os.path.isdir(path):
#         path_pickle = os.path.join(path, ''


def hash_path_md5(path):
    """
    Source: https://stackoverflow.com/a/22058673/1293027
    """
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()
