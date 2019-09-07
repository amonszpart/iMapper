import os


def listdir(d):
    """Get full paths to the files and folders in d.
    Args:
        d (str):
            Path of directory to scan.
    Returns:
        full_paths (genexp):
            The full paths to the entries in the directory d.
    """
    return (os.path.join(d, f) for f in os.listdir(d))


def makedirs_backed(d_dest, limit=200):
    """Creates a new directory, if it exists, it renames it to bak.xx"""
    if os.path.exists(d_dest):
        i = 0
        while i < limit:
            try:
                os.rename(d_dest, "%s.bak.%02d" % (d_dest, i))
                break
            except OSError:
                i += 1
        if i == limit:
            raise RuntimeError("Could not rename it more than %d" % i)
    os.makedirs(d_dest)
