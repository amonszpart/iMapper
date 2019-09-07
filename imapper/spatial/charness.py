import os

import numpy as np
import scipy.io

from imapper.logic.scenelet import Scenelet
from imapper.scenelet_fit.radial_histograms import \
    RadialHistogram, SquareHistogram
from imapper.util.my_pickle import hash_path_md5, pickle, pickle_load
from imapper.util.stealth_logging import lg, split_path


def parse_charness_histograms(dmat):
    assert 5 == len(dmat['pigraph_histogram_charness'].shape), \
        "Not 5D? %s" % repr(dmat['pigraph_histogram_charness'].shape)
    bins = np.transpose(dmat['pigraph_histogram_charness'], (4, 3, 0, 1, 2))
    names_scenelets = dmat['pigraph_scenelet_names']
    if 'pigraph_histogram_params' in dmat:
        hists = \
            dict((Scenelet.to_old_name(name_scenelet[0][0]),
                  SquareHistogram.from_mat(
                      params=dmat['pigraph_histogram_params'],
                      bins=bins[id_scenelet, :, :, :],
                      categories=dmat['categories']))
                 for id_scenelet, name_scenelet in enumerate(names_scenelets))
    else:
        hists = \
            dict((Scenelet.to_old_name(name_scenelet[0][0]),
                  RadialHistogram.from_mat(
                      angular_edges=dmat['angular_edges'],
                      radial_edges=dmat['radial_edges'],
                      bins=bins[id_scenelet, :, :, :],
                      categories=dmat['categories']))
                 for id_scenelet, name_scenelet in enumerate(names_scenelets))
    return hists


def read_charness_histograms(path):
    """
    ['pigraph_norm_factor', 'pigraph_histogram_params',
     'pigraph_histogram_charness', 'pigraph_pose_charness',
     'pigraph_scenelet_names', 'categories']
    """
    hash_mat_file_current = hash_path_md5(path)
    hists = None

    path_pickle = "%s.pickle" % path
    if os.path.exists(path_pickle):
        with open(path_pickle, 'rb') as f:
            # hists, hash_mat_file = pickle_load(f)
            tmp = pickle_load(f)
            if tmp[-1] != hash_mat_file_current:
                hists = None
                lg.warning("Hashes don't match, reloading hists")
            else:
                if len(tmp) == 3:
                    hists = tmp[1]  # pose_charness, hists, hash
                else:
                    hists = tmp[0]  # hists, hash
                lg.info("Loaded hists from\n\t%s!!!"
                        % split_path(path_pickle))

    if hists is None:
        dmat = scipy.io.loadmat(path)
        hists = parse_charness_histograms(dmat)
        with open(path_pickle, 'wb') as f:
            pickle.dump((hists, hash_mat_file_current), f, -1)
            lg.info("Saved hists to %s" % path_pickle)
    # print(hists.keys())
    # key = list(hists.keys())[0]
    # logging.info("key: %s, %s" % (key, hists[key].volume))
    return hists