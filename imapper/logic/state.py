import copy
import multiprocessing
import sys
from collections import Counter
from itertools import product
from math import floor

from imapper.util.stealth_logging import logging, lg

try:
    from imapper.visualization.plotting import plt
except ImportError:
    lg.error("Could not import plt")

import numpy as np
import shapely.geometry as geom
from builtins import range
from descartes.patch import PolygonPatch
from imapper.scenelet_fit.regular_grid_interpolator import RegularGridInterpolator
from scipy.stats import multivariate_normal, norm, rv_discrete, vonmises
from shapely.ops import cascaded_union

import imapper.logic.geometry as gm
from imapper.logic.geometry import rot_y
from imapper.logic.joints import Joint
from imapper.logic.mesh_OBJ import MeshOBJ, Scene
from imapper.visualization.vis_skeleton import VisSkeleton
try:
    from imapper.visualizer.visualizer import Visualizer
except ImportError:
    lg.error("Could not import vtk")
from imapper.config.conf import Conf
from imapper.util.timer import Timer


def rect_mask_from_obb(obbs, state):
    """Splats a rectangle with distance from boundary"""
    if isinstance(obbs, list):
        poly = cascaded_union(
            [
                geom.Polygon(obb.corners_3d_lower(
                    up_axis=(0., -1., 0.))[:, [0, 2]])
                for obb in obbs
            ])
    else:
        poly = geom.Polygon(obbs.corners_3d_lower()[:, [0, 2]])
        
    # logging.info(poly)
    grid_2d = state.get_grid_2d_grid()
    xs, zs = (ret.reshape(-1)
              for ret in grid_2d)
    ps = [(
        state.convert_grid_loc_to_world_pos(
           self=state, grid_loc=[xs[row], zs[row], 0.0]),
        [xs[row], zs[row]]
    ) for row in range(xs.size)]

    rect_mask = np.zeros((grid_2d[0].shape[0],
                          grid_2d[0].shape[1]),
                         dtype=np.float32)
    boundary = poly.boundary
    max_spatial_resolution = max(state.resolution[0], state.resolution[2])
    for p in ps:
        point = geom.Point(p[0][0], p[0][1])
        if poly.contains(point):
            distance = point.distance(boundary)
            score = min(1., distance + 1. - 3. * max_spatial_resolution)
            # logging.info("Distance: %s, maxres: %s, final: %s"
            #              % (distance, max_spatial_resolution, score))
            rect_mask[p[1][0], p[1][1]] = score

    # patch = PolygonPatch(poly, facecolor='b',
    #                      edgecolor='r', alpha=0.3)
    # plt.figure()
    # plt.imshow(rect_mask)
    # plt.scatter([p[0][0] for p in ps], [p[0][1] for p in ps])
    # ax = plt.gca()
    # ax.add_patch(patch)
    # plt.show()
    return rect_mask, poly


def rect_mask_from_obb_and_skeleton(obbs, state, skeleton):
    """Splats a rectangle with distance from skeleton"""
    if isinstance(obbs, list):
        poly = cascaded_union(
            [
                geom.Polygon(obb.corners_3d_lower(
                    up_axis=(0., -1., 0.))[:, [0, 2]])
                for obb in obbs
            ])
    else:
        poly = geom.Polygon(obbs.corners_3d_lower()[:, [0, 2]])

    pelv = skeleton._poses[:, [0, 2], Joint.PELV]
    # logging.info("\nPelv:\n%s\n(shape: %s)" % (pelv, repr(pelv.shape)))
    poly_skeleton = geom.LineString(pelv.tolist())
    # logging.info("polyskel: %s" % poly_skeleton)

    # logging.info(poly)
    grid_2d = state.get_grid(dim=2)
    xs, zs = (ret.reshape(-1)
              for ret in grid_2d)
    ps = [(
        state.convert_grid_loc_to_world_pos(
            self=state, grid_loc=[xs[row], zs[row], 0.0]),
        [xs[row], zs[row]]
    ) for row in range(xs.size)]

    rect_mask = np.zeros((grid_2d[0].shape[0],
                          grid_2d[0].shape[1]),
                         dtype=np.float32)
    # boundary = poly.boundary
    gauss = norm()
    max_spatial_resolution = max(state.resolution[0], state.resolution[2])
    for p in ps:
        point = geom.Point(p[0][0], p[0][1])
        if poly.contains(point):
            distance = point.distance(poly_skeleton)  # point.distance(boundary)
            gdist = norm(scale=.5).pdf(x=distance)
            # logging.info("gdist(%f): %f" % (distance, gdist))
            score = min(1., gdist + 1. - 3. * max_spatial_resolution)
            # score = min(1., 1. - distance + 1. - 3. * max_spatial_resolution)
            # logging.info("Distance: %s, maxres: %s, dist: %f, gdist: %f, final: %s"
            #              % (distance, max_spatial_resolution, distance, gdist, score))
            rect_mask[p[1][0], p[1][1]] = score

    if False:
        patch_poly = PolygonPatch(poly, facecolor='b',
                                  edgecolor='r', alpha=0.3)
        plt.figure()
        plt.imshow(rect_mask)
        plt.scatter([p[0][0] for p in ps], [p[0][1] for p in ps])
        ax = plt.gca()
        ax.add_patch(patch_poly)
        plot_line_shapeley(ax, poly_skeleton)
        plt.show()
    return rect_mask, poly


def get_orig_name(name_scenelet):
    return name_scenelet.partition('_aligned')[0]


def weigh_volume(volume, labels_to_lin_ids, dbn, scene):
    """Create a copy of the volume, where each category-slice is weighted
    by the dbn score of adding a NEW object of that category
    NOTE: new object only, does not score the current volume
    """

    n_samples = Conf.get().occurrence.n_samples
    out = volume.copy()
    assert out is not volume, "Deepcopy failed"

    curr_labels = {"%s_%d" % (label, v): ['1']
                   for label, v in scene.get_labels().items()}

    logging.info("[weigh_volume]: %s" % curr_labels)
    probs = np.zeros(len(labels_to_lin_ids))
    processes = []
    pool = multiprocessing.Pool()
    for label, lin_id in labels_to_lin_ids.items():
        labels_ = copy.deepcopy(curr_labels)

        label_ = next((l for l in labels_ if l.split('_')[0] == label), None)
        if label_ is not None:
            valence = int(label_.split('_')[1])
            # logging.info("Valence of %s: %d" % (label_, valence))
            query = {"%s_%d" % (label, valence + 1): ['1']}
        else:
            query = {"%s_%d" % (label, 1): ['1']}

        logging.info("Querying %s with evidence %s" % (query, labels_))
        processes.append((
            pool.apply_async(
               func=dbn.score,
               kwds={'query': query, 'evidence': labels_,
                     'n_samples': n_samples}
            ), label, query, labels_, lin_id
        ))
        # score_dbn = \
        #     dbn.score(query=query, evidence=labels_, n_samples=n_samples)
    pool.close()
    pool.join()

    for process, label, query, evidence, lin_id in processes:
        score_dbn = process.get()
        logging.info("Score_dbn[%s]: %f, query(%s), evidence(%s)"
                     % (label, score_dbn, query, evidence))
        probs[lin_id] = score_dbn

    probs_unnormed = {label: probs[lin_id]
                      for label, lin_id in labels_to_lin_ids.items()}
    # normalize
    sum_ = probs.sum()
    if sum_ > 0.:
        probs /= probs.sum()
    probs_dict = dict()
    for label, lin_id in labels_to_lin_ids.items():
        out[lin_id, :, :, :] *= probs[lin_id]
        probs_dict[label] = probs[lin_id]

    logging.info("\n".join(
       ["Score_dbn[%s]: %f" % (label, prob)
        for label, prob in probs_dict.items()]))
    return out, probs_dict, probs_unnormed


def weigh_volume_old(volume, labels_to_lin_ids, dbn, scene):
    """Create a copy of the volume, where each category-slice is weighted
    by the dbn score of adding a NEW object of that category
    NOTE: new object only, does not score the current volume
    """
    out = volume.copy()
    assert out is not volume, "Deepcopy failed"

    # get all larger valences
    curr_labels = \
        dict((label,
              sorted([valence
                      for valence in dbn.model.Vdata[label][u'vals']
                      if valence >= v],
                     reverse=True))
             for label, v in scene.get_labels().items())
    # don't zero them out
    # curr_labels.update((k, [0.]) for k in dbn.model.V if k not in curr_labels)

    print("[weigh_volume]: %s" % curr_labels)
    for label, lin_id in labels_to_lin_ids.items():
        labels_ = copy.deepcopy(curr_labels)

        # remove the smallest valence
        if label in labels_:
            labels_[label].pop(-1)
        else:
            # print(dir(dbn.model.Vdata[label]))
            labels_[label] = [valence
                              for valence in dbn.model.Vdata[label][u'vals']
                              if valence >= 1.]
        # assert len(labels_[label]), \
        #     "No valences left: %s (curr: %s)" % (labels_, curr_labels)

        score_dbn = 0.
        try:
            score_dbn = \
                dbn.score(fill=False, queries=labels_, evidence={},
                          verbose=False)
            print("Score_dbn[%s]: %f, query(%s)" % (label, score_dbn, labels_))
        except ValueError as e:
            print(e)
            print("Could not query dbn for %s, zeroing out!!" % labels_)

        out[lin_id, :, :] *= score_dbn
    return out


class State(object):
    """Stores the probability volume"""

    __TWO_PI = np.float32(2. * np.pi)

    def __init__(self, room, tr_ground_inv, res_theta,
                 resolution=Conf.get().volume.resolution,
                 ignore={u'floor', u'wall'},
                 sigma=np.float32(0.2),
                 charness_histograms=None):
        """Constructor
        :param resolution: Spatial resolution in meters.
        :param sigma: Gaussian lobe for each object
        """

        self.scenelets = []
        """List of scenelets and their weights, added in order"""
        self.room = room.astype(np.float32)
        self.room.flags.writeable = False
        """3x2 array with scene bounding box in 3D"""
        self.tr_ground_inv = tr_ground_inv
        if tr_ground_inv is not None:
            self.tr_ground_inv.flags.writeable = False
        """Inverse ground transform"""
        self.resolution = \
            np.array([resolution, resolution,  resolution,
                      np.float32(res_theta)], dtype=np.float32) \
                if isinstance(resolution, float) \
                   or isinstance(resolution, np.float32) \
                else np.array([resolution[0], resolution[1], resolution[2],
                               np.float32(res_theta)], dtype=np.float32)
        self.resolution.flags.writeable = False
        """
        Resolution (distance covered by one unit coordinate) 
        of the probability volume
        """

        self.charness_histograms = charness_histograms
        """Characteristicness histograms of scenelets"""

        self._changed = True
        """Dirty flag for volume"""

        self._volume = None
        """Probability volume representation"""
        self._samplers = None
        """Regular grid interpolators (dictionary, one per label) for _volume"""

        self._volume_normed = None
        """Probability volume representation normalized"""
        self._labels_to_lin_ids = None
        """Mapping from categories to the first dimension in the volume"""
        self._lin_ids_to_labels = None
        """Reverse mapping of same"""
        self._region_masks = None
        """Volume modifiers based on objects in the scene"""
        self._grid_points = dict()
        """World coordinates and grid coordinates for given volume"""
        self._grid_points_shapely = None

        self._sigma = sigma
        """Gaussian lobe for each object in the volume"""
        self._labels_unique = set()
        """Unique labels"""
        self._ignore = ignore

    def set_changed(self, changed):
        if changed:
            self._volume = None
            self._volume_normed = None
            self._labels_to_lin_ids = None
            self._lin_ids_to_labels = None
            self._region_masks = None
            self._grid_points = dict()
            self._grid_points_shapely = None
            self._samplers = dict()
        self._changed = changed

    @property
    def volume_normed(self):
        return self._volume_normed

    @property
    def region_masks(self):
        return self._region_masks

    def add_scenelet(self, sclt, weight=1.):
        if sclt in {sclt_weight[0] for sclt_weight in self.scenelets}:
            return False
        cpy = copy.deepcopy(sclt)
        self.scenelets.append((cpy, weight))
        self._labels_unique = \
            self._labels_unique.union(
                set(cpy.get_labels(ignore=self._ignore).keys()))

        self.set_changed(True)
        return True

    def get_labels_to_lin_ids(self):
        if self._labels_to_lin_ids is None or not len(self._labels_to_lin_ids):
            self._labels_to_lin_ids = self._calc_labels_to_lin_ids()
        return self._labels_to_lin_ids

    def _calc_labels_to_lin_ids(self):
        labels = set(self._labels_unique).difference(self._ignore)
        return dict((label, i) for i, label in enumerate(labels))

    def get_lin_ids_to_labels(self):
        if self._lin_ids_to_labels is None:
            self._lin_ids_to_labels = \
                dict((value, key)
                     for key, value in self.get_labels_to_lin_ids().items())
        return self._lin_ids_to_labels

    @staticmethod
    def _create_cov(sigma):
        if isinstance(sigma, float) or len(sigma) == 1:
            return [[sigma, 0., 0.], [0., sigma, 0.], [0., 0., sigma]]
        elif len(sigma) == 3:
            return [[sigma[0], 0., 0.], [0., sigma[1], 0.], [0., 0., sigma[1]]]
        elif len(sigma) == 9:
            return sigma
        else:
            raise ValueError("Can't parse shape of %s" % sigma)

    def angular_splat(self, slice_2d, angle, sigma):
        """Lift a 2d spatial slice to a 3d slice with sigma falloff"""
        # weigh rotation dimension
        # vonmises: 1/kappa = sigma^2
        kappa = 1. / (sigma * sigma)
        thetas = self.get_angles()
            # np.arange(0., self.__TWO_PI-self.resolution[3],
            #           self.resolution[3])
        assert np.allclose(self.get_angles(), thetas), \
            "Not close:\n%s,\n%s" % (self.get_angles(), thetas)
        loc_vonmises = angle - np.pi  # centered around 0, not pi
        pdf = vonmises.pdf(thetas, kappa, loc=loc_vonmises)
        pdf = np.roll(pdf, pdf.shape[0]//2, axis=0)

        p = np.tile(slice_2d[:, :, np.newaxis],
                    (1, 1, self._volume.shape[3]))
        return p * pdf

    def splat_scenelet(self, ob, sclt):
        c = ob.get_centroid()
        angle = ob.get_angle(positive_only=True)
        obbs = [part.obb for part in ob.parts.values()]
        # rect_mask, _ = \
        #     rect_mask_from_obb(obbs, self)
        rect_mask, _ = \
            rect_mask_from_obb_and_skeleton(
                obbs, self, sclt.skeleton)

        # get middle frame skeleton local coordinate frame in 2D
        sclt_frames = sclt.skeleton.get_frames()
        mid_frame_id = sclt_frames[len(sclt_frames) // 2]
        fw = sclt.skeleton.get_forward(mid_frame_id,
                                       estimate_ok=False)
        # homogeneous 2x3
        transform = np.identity(3, dtype=np.float32)
        # x is forward
        transform[:2, 0] = gm.normalized(fw[[0, 2]])
        # z is orthogonal: orthogonal to (x, y) is (-y, x)
        transform[0, 1] = -transform[1, 0]
        transform[1, 1] = transform[0, 0]
        # translation is the pelvis at the time
        transform[:2, 2] = \
            sclt.skeleton.get_joint_3d(
                Joint.PELV, mid_frame_id)[[0, 2]]

        # inverse transform
        inv_transform = np.linalg.inv(transform)

        # assemble local space polygon
        # rect comes as a list of row vectors,
        # multiplication therefore has to happen on the transposed
        # and then polygon wants row-vectors again, so one more .T
        poly_local = cascaded_union([
            geom.Polygon(
                np.dot(inv_transform[:2, :2],  # rotation part
                       obb.corners_3d_lower()[:, [0, 2]].T).T
                + inv_transform[:2, 2]  # translation part
            )
            for obb in obbs])  # union of all object parts

        # transform name
        name_scenelet_orig = get_orig_name(sclt.name_scenelet)
        # get histogram
        hist = \
            self.charness_histograms[(sclt.name_scene,
                                      name_scenelet_orig)]
        # compute object weight based on area overlap
        weight_hist = hist.get_weight(ob.label, poly_local)

        # debug vis
        if False and (('ahouse_mati5_2014-05-16-19-57-44' in sclt.name_scene) \
                          and ('scenelet_24' in sclt.name_scenelet)):
            # and ob.label == u'couch':
            vis = Visualizer()
            vis.add_coords()
            for oid, obb in enumerate(obbs):
                vis.add_mesh(MeshOBJ.from_obb(obb), "obb%d" % oid)
            VisSkeleton.vis_skeleton(
                vis, sclt.skeleton.get_pose(mid_frame_id), "skel")
            vis.show()
            hist.plot(show=False,
                      polys_show={ob.label:
                                      [(poly_local, weight_hist)]})

            logging.info("Debugshow: %s"
                         % repr(sclt.name_scene,
                                sclt.name_scenelet, ob.label))
            plt.show()
            plt.close()
        else:
            logging.info(
                "%s" % repr((sclt.name_scene, sclt.name_scenelet,
                             ob.label)))

        # logging.info("Weight [%s] is %f"
        #              % (sclt.name_scenelet, weight_hist))

        assert rect_mask.shape == \
               (self._volume.shape[1], self._volume.shape[2]), \
            "Wrong shape: %s" % repr(rect_mask.shape)

        # weigh rotation dimension
        # vonmises: 1/kappa = sigma^2
        kappa = 1. / (self._sigma[2] * self._sigma[2])
        thetas = \
            np.arange(0., self.__TWO_PI-self.resolution[3],
                      self.resolution[3])
        loc_vonmises = angle - np.pi  # centered around 0, not pi
        pdf = vonmises.pdf(thetas, kappa, loc=loc_vonmises)
        # logging.info("pdf: %s" % pdf)
        pdf = np.roll(pdf, pdf.shape[0]//2, axis=0)

        if False and (ob.label == u'couch'):
            plt.figure()
            ax = plt.gca()
            # x = np.linspace(vonmises.ppf(0.01, kappa),
            #                 vonmises.ppf(0.99, kappa), 100)
            logging.debug("angle: %s, thetas: %s"
                          % (np.rad2deg(angle),
                             thetas))
            ax.plot(thetas, pdf,
                    'r-', lw=5, alpha=0.6, label='vonmises pdf')
            plt.show()

        p = np.tile(rect_mask[:, :, np.newaxis],
                    (1, 1, self._volume.shape[3]))
        tmp_shape = p.shape  # debug shape

        p *= pdf
        assert p.shape == tmp_shape, \
            "Wrong shape: %s" % repr(p.shape)

        if False and ob.label == u'couch':
            for part_id, part in ob.parts.items():
                vis.add_mesh(MeshOBJ.from_obb(part.obb),
                             "ob_%d_%s_%s" % (oid, ob.label, part.label))
            vis.add_arrows(
                centroid=np.squeeze(ob._largest_part().obb.centroid),
                transform= ob.get_transform(),
                name="ob_%d_%s_%s_coords" % (part_id, ob.label, part.label))
            logging.info(
                "Debugshow: %s"
                % repr((sclt.name_scene, sclt.name_scenelet,
                        ob.label, oid)))
            vis.show()
            vis.remove_all_actors(prefixes={"ob_"})

        # use histogram weight
        p *= weight_hist
        # logging.debug("p.shape: %s" % repr(p.shape))

        return p

    def splat_histogram(self, scenelet, shape, id_stat=None):
        """

        :param scenelet:
        :param shape: Shape of volume to mirror.
        :return:
        """
        if id_stat is not None and not id_stat[0] % 10:
            lg.info("Starting %d/%d" % (id_stat[0], id_stat[1]))
        # lg.info("Starting %s" % scenelet.name_scenelet)
        transform = \
            scenelet.skeleton.get_transform_from_forward(dim=2, frame_id=-1)

        # get middle frame skeleton local coordinate frame in 2D
        # sclt_frames = scenelet.skeleton.get_frames()
        # sclt_frames[len(sclt_frames) // 2]
        # mid_frame_id = scenelet.skeleton.get_representative_frame()
        # fw = scenelet.skeleton.get_forward(mid_frame_id,
        #                                    estimate_ok=False)
        # inverse transform
        inv_transform = np.linalg.inv(transform).astype('f4')

        # transform name
        name_scenelet_orig = get_orig_name(scenelet.name_scenelet)
        # get histogram
        hist = self.charness_histograms[(scenelet.name_scene,
                                         name_scenelet_orig)]
        assert hist is not None, "Could not get histogram by (%s, %s)" \
                                 % (scenelet.name_scene, name_scenelet_orig)

        slice_ = np.zeros(shape=shape, dtype='f8')
        ps = self.get_grid_points_np(dim=3, clone=True)
        assert ps.dtype == np.float32, "Wrong type: %s" % ps.dtype
        assert ps.shape[1] == 6, "Wrong shape: %s" % repr(ps.shape)
        # ps_old = ps.copy()
        # logging.info("First point is %s" % ps[0, :])
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # axes = axes.ravel()
        # axes[0].scatter(ps[:, 0], ps[:, 1], c='b')
        ps[:, :2] = np.dot(inv_transform[:2, :2], ps[:, :2].T).T \
                    + inv_transform[:2, 2]
        assert ps.dtype == np.float32, "Wrong type: %s" % ps.dtype
        angle_diff = np.arctan2(transform[1, 0], transform[0, 0])
        # logging.info("Angle diff of %s is %s"
        #              % (transform[:2, 0], np.rad2deg(angle_diff)))
        # - -atan2 ==> + atan2
        ps[:, 2] += np.arctan2(transform[1, 0], transform[0, 0])
        assert ps.dtype == np.float32, "Wrong type: %s" % ps.dtype
        for row in range(ps.shape[0]):
            if ps[row, 2] > self.__TWO_PI:
                ps[row, 2] -= self.__TWO_PI
            elif ps[row, 2] < np.float32(0.):
                ps[row, 2] += self.__TWO_PI
        assert ps.dtype == np.float32, "Wrong type: %s" % ps.dtype
        # logging.info("First point is now %s" % ps[0, :])
        # axes[1].scatter(ps[:, 0], ps[:, 1], c='g')
        # plt.show()
        # assert ps.shape == ps_old.shape, \
        #     "No: %s %s" % (ps.shape, ps_old.shape)
        labels_to_lin_ids = self.get_labels_to_lin_ids()
        # TODO: enlarge grid to reach edges

        # def f(r, cat):
        #     lin_id = labels_to_lin_ids[cat]
        #     try:
        #         if hist.contains(r[:3], cat=cat):
        #             slice_[lin_id,
        #                    np.int32(r[3]), np.int32(r[4]), np.int32(r[5])] = \
        #                 hist.get_value(r[:3], cat=cat)
        #     except KeyError:  # histogram does not have cat
        #         pass
        #
        #     return None

        gen = ((label, lin_id)
               for label, lin_id in self.get_labels_to_lin_ids().items()
               if hist.has_label(label))
        l_contains = hist.contains
        l_get_value = hist.get_value
        for cat, lin_id in gen:
            # rows_gt = []  # record which were contained
            # with Timer("v0", verbose=True):
            #     for row in range(ps.shape[0]):
            #         try:
            #             r = ps[row, :]
            #             if l_contains(r[:3], cat=cat):
            #                 slice_[lin_id,
            #                        np.int32(r[3]), np.int32(r[4]),
            #                        np.int32(r[5])] = \
            #                     l_get_value(r[:3], cat=cat)
            #
            #                 # debug
            #                 rows_gt.append(row)
            #         except KeyError:  # histogram does not have cat
            #             logging.debug("This should not happen")
            #             pass
            #
            # _slice_gt = slice_[lin_id, :, :, :].copy()
            # with Timer("v1", verbose=True):
            grid = hist._samplers[cat].grid
            rows = np.logical_and(
               np.logical_and(
                  ps[:, 0] >= grid[0][0],
                  ps[:, 0] <= grid[0][-1]
               ),
               np.logical_and(
                  ps[:, 1] >= grid[1][0],
                  ps[:, 1] <= grid[1][-1]
               )
            ).nonzero()[0]

            # if not np.array_equal(rows_gt, rows):
            #     diff = set(rows).difference(set(rows_gt))
            #     for d in diff:
            #         print(
            #             "%d, %s, edges: %s"
            #             % (d,
            #                ps[d, :],
            #                (
            #                    hist.edges[0][0], hist.edges[0][-1],
            #                    hist.edges[1][0], hist.edges[1][-1],
            #                    hist.edges[2][0], hist.edges[2][-1])
            #                )
            #         )
            #         break
            # assert np.array_equal(rows_gt, rows), \
            #     "No: %s" % set(rows).difference(set(rows_gt))

            indices = ps[rows, :]
            values = l_get_value(indices[:, :3], label=cat)
            slice_[lin_id,
                   indices[:, 3].astype(int),
                   indices[:, 4].astype(int),
                   indices[:, 5].astype(int)] = values

            # assert np.allclose(slice_[lin_id, :, :, :], _slice_gt), "no"
                # for row in rows:  # range(ps.shape[0]):
                #     try:
                #         r = ps[row, :]
                        # if l_contains(r[:3], cat=cat):
                        #     slice_[lin_id,
                        #            np.int32(r[3]), np.int32(r[4]),
                        #            np.int32(r[5])] = \
                        #         l_get_value(r[:3], cat=cat)
                    # except KeyError:  # histogram does not have cat
                    #     logging.debug("This should not happen")
                    #     pass

        # [f(ps[row, :], cat)
        #  for cat, lin_id in self.get_labels_to_lin_ids().items()
        #  for row in range(ps.shape[0])]

        # for cat, lin_id in self.get_labels_to_lin_ids().items():
        #     for row in range(ps.shape[0]):
        #         f(ps[row, :], cat)
            # np.apply_along_axis(f, axis=1, arr=ps, cat=cat)
        # logging.info("Sum(slice_2d): %f" % np.sum(slice_2d))
        # slice_3d = self.angular_splat(slice_2d, angle,
        #                               sigma=self._sigma[2])

        return slice_

    def set_volume(self, volume, keep_region_masks, labels_to_lin_ids=None):
        self._volume = volume
        self._volume.flags.writeable = False
        sum = self._volume.sum()
        if sum > 0.:
            self._volume_normed = self._volume / sum
        else:
            self._volume_normed = self._volume.copy()
        self._volume_normed.flags.writeable = False

        if labels_to_lin_ids is not None:
            self._labels_to_lin_ids = labels_to_lin_ids
            self._lin_ids_to_labels = None
            self.get_lin_ids_to_labels()

        if not keep_region_masks:
            self._region_masks = \
                dict((label,
                      np.ones(self._volume[lin_id].shape, dtype=np.float32))
                     for label, lin_id in self._labels_to_lin_ids.items())

        if self._samplers is not None:
            self._samplers.clear()

        self.set_changed(False)

    def get_volume(self, labels_to_lin_ids_arg=None):
        if not self._changed:
            assert labels_to_lin_ids_arg is None, \
                "labels_to_lin_ids argument should only be provided, " \
                "when creating the volume."
            return self._volume, self.get_labels_to_lin_ids()

        volume, labels_to_lin_ids = self._create_volume(
           labels_to_lin_ids_arg=labels_to_lin_ids_arg)
        self._labels_to_lin_ids = labels_to_lin_ids
        self.set_volume(volume, keep_region_masks=False)
        return self._volume, labels_to_lin_ids

    @property
    def volume(self):
        if not self._changed:
            return self._volume
        else:
            return self.get_volume()[0]

    def _create_volume(self, labels_to_lin_ids_arg=None):
        with_confidence = Conf.get().volume.with_confidence
        if not with_confidence:
            lg.warning("--------NOT USING CONFIDENCE-------")
        with_density = Conf.get().volume.with_density
        if not with_density:
            lg.warning("--------NOT USING DENSITY-------")
        grid_size = self.get_grid_size()
        labels_to_lin_ids = self._calc_labels_to_lin_ids() \
            if labels_to_lin_ids_arg is None \
            else copy.deepcopy(labels_to_lin_ids_arg)
        volume = \
            np.zeros(
                shape=(len(labels_to_lin_ids), grid_size[0], grid_size[2], grid_size[3]),
                dtype=float)
        mgrid = np.mgrid
        mt = False
        # sys.stderr.write("TODO:: Re-enable mt for volume splatting!\n")
        if mt:  # multi-threaded
            pool = multiprocessing.Pool()
            processes = []
            for sclt, weight in self.scenelets:
                density = sclt.density if with_density else 1.
                confidence = sclt.confidence if with_confidence else 1.
                processes.append(
                   (
                       sclt.charness * weight * confidence / density,
                       pool.apply_async(
                          func=self.splat_histogram,
                          args=[sclt, volume.shape,
                                (len(processes), len(self.scenelets))]
                       )
                   )
                )
            pool.close()
            pool.join()
            for weight, process in processes:
                volume += weight * process.get()
        else:  # non-mt
            ii = 0
            for sclt, weight in self.scenelets:
                density = sclt.density if with_density else 1.
                confidence = sclt.confidence if with_confidence else 1.
                volume += \
                    sclt.charness * weight * confidence / density \
                    * self.splat_histogram(sclt, volume.shape,
                                           (ii, len(self.scenelets)))
                ii += 1
        return volume, labels_to_lin_ids

    def get_volume_current(self, scene, dbn):
        """
        Creates a volume from _volume_normed by weighing each label slice
        the probability of placing a next object of that label
        and by masking using _region_masks.
        :return : per-label layer weights
        (normalized probabilities for new object of that category)
        """
        assert not self._changed and self._volume is not None, \
            "Expected get_volume to be called first"
        if dbn is not None:
            weighed_volume, probabilities, probabilities_unnormed = \
                weigh_volume(self._volume_normed,
                             self._labels_to_lin_ids, dbn, scene)
        else:
            weighed_volume = self.volume_normed.copy()
            labels_to_lin_ids = self.get_labels_to_lin_ids()
            probabilities_unnormed = {label: np.random.uniform()
                                      for label in labels_to_lin_ids}
            sm = np.sum([prob for prob in probabilities_unnormed.values()])
            probabilities = {label: prob/sm
                             for label, prob in probabilities_unnormed.items()}

        for label, lin_id in self._labels_to_lin_ids.items():
            weighed_volume[lin_id] \
                *= self._region_masks[label].astype(weighed_volume.dtype)

        return weighed_volume, probabilities, probabilities_unnormed

    @staticmethod
    def create_sampler_points(grid_size, room, resolution):
        assert len(grid_size) == 4, "Expected 4d grid size"
        grid_size = copy.deepcopy(grid_size)
        grid_size.pop(1)  # don't need second dimension (y), only x, z, theta

        # construct sampler grid points
        points = [[], [], []]
        for ci in range(len(grid_size)):
            for p_i in range(grid_size[ci]):
                p = [0, 0, 0]
                p[ci] = p_i
                points[ci].append(
                    State.convert_grid_loc_to_world_pos(
                        grid_loc=p, room=room, resolution=resolution,
                        dtype=np.float32
                    )[ci]
                )

    @staticmethod
    def get_sampler_points(self=None, grid_size=None, room=None,
                           resolution=None, dim=3):
        if self is not None:
            assert isinstance(self, State), "Self needs to be State"
            if grid_size is None:
                grid_size = self.get_grid_size()
            if room is None:
                room = self.room
            if resolution is None:
                resolution = self.resolution
        else:
            assert room is not None and grid_size is not None \
                   and resolution is not None, \
                "Need all these arguments, if called static..."
        if grid_size.size == 4:
            grid_size = np.take(grid_size, (0, 2, 3))
        assert grid_size.size == 3, "Wrong size: %s" % repr(grid_size.shape)
        # construct sampler grid points
        points = [[], [], []]
        for ci in range(len(grid_size)):
            for p_i in range(grid_size[ci]):
                p = [0, 0, 0]
                p[ci] = p_i
                points[ci].append(
                    State.convert_grid_loc_to_world_pos(
                        grid_loc=p, room=room, resolution=resolution,
                        dtype=np.float32
                    )[ci]
                )

        return points[:dim]

    @staticmethod
    def create_samplers(volume, labels_to_lin_ids, grid_size, room, resolution):
        """
        Create a sampler from a volume
        (volume, normed volume, or weighed volume). Does NOT store it.
        """
        # grid_size = self.get_grid_size().tolist()
        # assert len(grid_size) == 4, "Expected 4d grid size"
        # grid_size = copy.deepcopy(grid_size)
        # grid_size.pop(1)  # don't need second dimension (y), only x, z, theta
        #
        # # construct sampler grid points
        # points = [[], [], []]
        # for ci in range(len(grid_size)):
        #     for p_i in range(grid_size[ci]):
        #         p = [0, 0, 0]
        #         p[ci] = p_i
        #         points[ci].append(
        #             State.convert_grid_loc_to_world_pos(
        #                 grid_loc=p, room=room, resolution=resolution,
        #                 dtype=np.float32
        #             )[ci]
        #         )
        points = State.get_sampler_points(
            grid_size=grid_size, room=room, resolution=resolution, dim=3)

        samplers = {
            label: RegularGridInterpolator(
                points=points, values=volume[lin_id], fill_value=0.,
                bounds_error=False)
            for label, lin_id in labels_to_lin_ids.items()}

        # for label, lin_id in labels_to_lin_ids.items():
        #     samplers[label] = \

        # samplers = dict()
        # for label, lin_id in labels_to_lin_ids.items():
        #     samplers[label] = \
        #         RegularGridInterpolator(
        #             points=points, values=volume[lin_id], fill_value=0.,
        #             bounds_error=False)

        return samplers

    def count_label(self, label=None):
        """Count, how many examples contributed to a category slice of 
        the volume, or all slices, if label is None
        """
        labels = [ob.label for sclt, weight in self.scenelets
                  for ob in sclt.objects.values()
                  if ob.label not in self._ignore]
        counter = Counter(labels)
        if label is None:
            return dict(counter)
        else:
            return counter[label]

    def get_grid_size(self):
        out = (self.room[:, 1] - self.room[:, 0]) / self.resolution[:3]
        out = np.append(out, int(round(self.__TWO_PI / self.resolution[3])))
        assert abs(self.resolution[3] * out[-1] - self.__TWO_PI) < 1.e-3, \
            "Wrong grid size: %s (res: %s, div: %s)" \
            % (out, self.resolution[3], self.__TWO_PI / self.resolution[3])
        assert out.shape == (4,), "No: s" % repr(out.shape)
        return np.ceil(out).astype(np.int32)

    @staticmethod
    def convert_grid_loc_to_world_pos(self=None, grid_loc=None, room=None,
                                      resolution=None, dtype=np.float32):
        if self is not None:
            assert isinstance(self, State), "Self has to be State"
        assert grid_loc is not None, \
            "grid_loc parameter is required"

        if self is None:
            assert room is not None and resolution is not None, \
                "Static method needs all arguments!"
        else:
            room = self.room
            resolution = self.resolution

        #     return State.convert_grid_loc_to_world_pos(grid_loc, )
        assert resolution.size == 4, \
            "Need 3+1D resolution, 3 spatial and one angular..." \
            "yes, resolution[1] is not used for now..."
        assert room.shape == (3, 2), \
            "Need room to be 3x2, each row is min-max for a spatial dimension"

        if room.dtype != dtype:
            sys.stderr.write(
                "Room comes in different type: %s vs %s"
                % (room.dtype, dtype))
        if resolution.dtype != dtype:
            sys.stderr.write(
                "Resolution comes in different type: %s vs %s"
                % (resolution.dtype, dtype))

        # convert list/tuple to numpy
        if isinstance(grid_loc, (list, tuple)):
            grid_loc = np.asarray(grid_loc, dtype=int)
        else:
            assert grid_loc.dtype == np.int32, \
                "Expected integer grid locations: %s" % grid_loc.dtype

        # single point or multiple queries?
        dim = len(grid_loc.shape)

        if dim == 1:  # single point
            assert grid_loc.shape[0] == 3, \
                "Expected nx3 integers: %s" % repr(grid_loc)

            return np.asarray(
                [
                    room[0, 0] + resolution[0] * (grid_loc[0] + 0.5),
                    room[2, 0] + resolution[2] * (grid_loc[1] + 0.5),
                    0. + resolution[3] * grid_loc[2]
                ],
                dtype=dtype)
        elif dim == 2:  # array query
            assert grid_loc.shape[1] == 3 or grid_loc.shape[1] == 2, \
                "Expected nx(2 or 3) integers: %s" % repr(grid_loc)

            xs = [room[0, 0] + resolution[0] * (dtype(row) + dtype(0.5))
                  for row in grid_loc[:, 0]]
            zs = [room[2, 0] + resolution[2] * (dtype(col) + dtype(0.5))
                  for col in grid_loc[:, 1]]
            if grid_loc.shape[1] == 3:
                thetas = [dtype(0.) + resolution[3] * dtype(id_theta)
                          for id_theta in grid_loc[:, 2]]
                out = np.vstack((xs, zs, thetas)).T
            else:
                out = np.vstack((xs, zs)).T

            assert out.shape == grid_loc.shape, \
                "wrong shape: %s" % repr(out.shape)
            assert out.dtype == dtype, \
                "Wrong type: %s vs %s" % (out.dtype, dtype)

            return out
        else:
            raise RuntimeError("Wrong shape dim, need 2d")

    # def convert_grid_loc_to_world_pos(self, grid_loc, dtype=np.float32):
    #     return State.convert_grid_loc_to_world_pos(grid_loc, )

        # if isinstance(grid_loc, (list, tuple)):
        #     grid_loc = np.asarray(grid_loc, dtype=int)
        # dim = len(grid_loc.shape)
        # if dim == 1:
        #     assert grid_loc.shape[0] == 3, \
        #         "Expected nx3 integers: %s" % repr(grid_loc)
        #     return np.asarray(
        #         [self.room[0, 0] + self.resolution[0] * (grid_loc[0] + 0.5),
        #          self.room[2, 0] + self.resolution[2] * (grid_loc[1] + 0.5),
        #          0. + self.resolution[3] * grid_loc[2]],
        #         dtype=dtype)
        # elif dim == 2:
        #     assert grid_loc.shape[1] == 3 or grid_loc.shape[1] == 2, \
        #         "Expected nx(2 or 3) integers: %s" % repr(grid_loc)
        #     xs = [dtype(self.room[0, 0] + self.resolution[0] * (row + 0.5))
        #           for row in grid_loc[:, 0]]
        #     zs = [dtype(self.room[2, 0] + self.resolution[2] * (col + 0.5))
        #           for col in grid_loc[:, 1]]
        #     if grid_loc.shape[1] == 3:
        #         thetas = [dtype(0. + self.resolution[3] * id_theta)
        #                   for id_theta in grid_loc[:, 2]]
        #         # out = np.asarray([xs, zs, thetas], dtype=dtype)
        #         out = np.vstack((xs, zs, thetas)).T
        #     else:
        #         out = np.vstack((xs, zs)).T
        #
        #     assert out.shape == grid_loc.shape, \
        #         "wrong shape: %s" % repr(out.shape)
        #     return out
        # else:
        #     raise RuntimeError("Wrong shape dim, need 2d")

    def convert_world_pos_to_grid_loc(self, pos, resolution=None):
        if resolution is None:
            resolution = self.resolution

        with_theta = True
        if len(pos) == 4:
            x, z, theta = pos[0], pos[2], pos[3]
        elif len(pos) == 3:
            x, z, theta = pos[0], pos[1], pos[2]
        elif len(pos) == 2:
            x, z = pos[0], pos[1]
            with_theta = False
        else:
            raise RuntimeError("Could not parse pos: %s" % pos)

        if with_theta:
            out = [int(floor((x - self.room[0, 0]) / resolution[0])),
                   int(floor((z - self.room[2, 0]) / resolution[2])),
                   int(round(theta / resolution[3]))]
            # angle layer turns around (round can go up)
            if out[2] == self._volume.shape[3]:
                out[2] = 0
        else:
            out = [int(floor((x - self.room[0, 0]) / resolution[0])),
                   int(floor((z - self.room[2, 0]) / resolution[2]))]

        return out

    @staticmethod
    def convert_world_pos_to_grid_loc_static(pos, resolution, room):
        # pos
        if isinstance(pos, (list, tuple)):
            pos = np.array(pos)
            if pos.ndim < 2:
                pos[None, :].reshape((pos.size // 2, 2))
            pos.flags.writeable = False
        else:
            pos = pos.reshape((-1, 2))
            pos.flags.writeable = False

        # resolution
        if isinstance(resolution, list):
            resolution = np.array(resolution)
        elif isinstance(resolution, (float, np.float32)):
            resolution = np.array([resolution, resolution])
        if resolution.size == 3:
            resolution = resolution[[0, 2]]
        else:
            assert resolution.size == 2, \
                "Invalid resolution: %s" % resolution
        resolution.flags.writeable = False

        # room
        if isinstance(room, (tuple, list)):
            room = np.array(room)
        if room.ndim == 1:
            room = room.reshape((-1, 2))
        if room.shape[0] > 2:
            assert room.size == 6, "Room can't be more, than min, max 3D"
            room = room[[0, 2], :]
        assert room.shape == (2, 2), "? %s" % room
        room.flags.writeable = False

        assert (pos.ndim == 1 and len(pos) == 2) or pos.shape[1] == 2, \
            "Assumed points in rows of pos, " \
            "if not, make sure to disambiguate (x, z, theta) and (x, y, z)"
        tmp = np.floor((pos - room[:, 0]) / resolution).astype(np.int32)
        return np.squeeze(tmp, axis=0)

    def get_angles(self):
        """Gets all angles corresponding to layers in the volume"""
        grid = self.get_grid_size()
        return [
            State.convert_grid_loc_to_world_pos(
                self=self, grid_loc=[0, 0, id_theta])[2]
            for id_theta in range(grid[3])
        ]

    def get_grid_2d_world(self):
        raise RuntimeError("TODO: Update to 3D?...")
        x, z = np.mgrid[
               self.room[0, 0]:self.room[0, 1]:self.resolution[0],
               self.room[2, 0]:self.room[2, 1]:self.resolution[2]]
        return x, z

    def get_grid(self, dim=2):
        grid_size = self.get_grid_size()
        # logging.debug("Grid size:%s" % grid_size)
        if dim == 2:
            return np.mgrid[
                   0:grid_size[0], 0:grid_size[2]].astype('i4')
        elif dim == 3:
            return np.mgrid[
                   0:grid_size[0], 0:grid_size[2], 0:grid_size[3]].astype('i4')
        else:
            raise RuntimeError("Can't do dim: %d" % dim)

    def get_object_theta_id(self, obj):
        """Gets closest angle to input object's angle in volume"""
        angles_volume = self.get_angles()

        # obb = obj.get_obb()
        angle_obj = obj.get_angle(positive_only=True)
        id_theta = \
            min(((id_theta_, angle_)
                 for id_theta_, angle_ in enumerate(angles_volume)),
                key=lambda pair: abs(pair[1] - angle_obj))[0]
        return id_theta

    def get_grid_points_np(self, dim, clone):
        """
        Gives [[world_x, world_z], [grid_x, grid_z],
                  ...
                 ]
        """
        assert (dim == 2) or (dim == 3), "No! %s" % dim
        try:
            return copy.deepcopy(self._grid_points[dim]) \
                if clone else \
                self._grid_points[dim]
        except KeyError:
            xs_zs_ts = np.array([ret.reshape(-1)
                                 for ret in self.get_grid(dim)],
                                dtype='i4')
            ps_world = State.convert_grid_loc_to_world_pos(
                self=self, grid_loc=xs_zs_ts.T)
            self._grid_points[dim] = np.hstack(
                (ps_world, xs_zs_ts.T.astype('f4')))
            self._grid_points[dim].flags.writeable = False
            ps2 = self._grid_points[dim]
            assert ps2.shape == (xs_zs_ts[0].size, dim*2), \
                "No: %s %s" % (ps2.shape, (xs_zs_ts[0].size, dim*2))
            return copy.deepcopy(self._grid_points[dim]) \
                if clone \
                else self._grid_points[dim]

    def get_grid_points_shapely(self, clone):
        if self._grid_points_shapely is None:
            self._grid_points_shapely = \
                tuple(geom.Point(p[0], p[1])
                      for p in self.get_grid_points_np(dim=2, clone=False))
        return copy.deepcopy(self._grid_points_shapely) if clone \
            else self._grid_points_shapely

    def get_room_center(self) -> np.ndarray:
        """The center of the room, i.e., the center
        of the 2D grid in world coordinates.
        """
        return self.room[:, 0] + self.get_room_size() / np.float32(2.)

    def get_room_size(self) -> np.ndarray:
        """The size of the room in world coordinates."""
        return self.room[:, 1] - self.room[:, 0]


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

if sys.version_info[0] < 3:
    import copy_reg
    import types
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class DiscreteProbGen(rv_discrete):
    def __init__(self, evidence, name='DP'):
        super(DiscreteProbGen, self).__init__(name=name, a=0, b=evidence.size)
        # self.evidence = (evidence / np.sum(evidence)).ravel()
        self.evidence = (evidence / np.sum(evidence)).ravel()

    def _pmf(self, x):
        # print("x(%s): %s" % (repr(x.shape), x))
        # print(y)
        # print("y(%s): %s" % (repr(y.shape), y))
        # print("x: %s" % x)
        # print("type(x): %s" % type(x))
        # print("evid.shape: %s" % repr(self.evidence.shape))
        return self.evidence[x.astype(int)]


_label_to_int = {
    u'couch': 0,
    u'table': 1,
    u'whiteboard': 2,
    u'bookshelf': 3,
    u'tvstand': 4,
    u'tv': 5,
    u'chair': 6,
    u'plant': 7,
    u'stool': 8
}


def render(state, labels, scene_objects):
    raise DeprecationWarning("Deprecated")
    # convert set to non-unique list of labels to consume
    _labels = []
    for label in labels:
        for i in range(int(labels[label])):
            _labels.append(label)
    _labels = sorted(_labels, key=lambda l: _label_to_int[l])
    # print("_labels: %s" % _labels)

    # cache scene bounding box
    room = state.room

    volume, lin_ids = state.get_volume()
    # scenelet = Scenelet()
    scene = Scene(name="Rendered")
    iteration = -1
    while len(_labels) and iteration < 20:
        iteration += 1
        label = np.random.choice(_labels)
        # print("label is: %s" % label)
        try:
            evidence = volume[lin_ids[label], :, :]
        except KeyError:
            print("Don't have %s in the prob volume, skipping" % label)
            _labels.remove(label)
            continue
        if label not in scene_objects:
            print("No such object in scenelet...")
            continue

        dp = DiscreteProbGen(evidence, name='dp')
        not_found = True
        while not_found:
            try:
                rvs = dp.rvs(size=1)
                samples = np.unravel_index(rvs, evidence.shape)
            except ValueError as e:
                print("ValueError: %s" % e)
                print("rvs: %s, shape: %s" % (rvs, repr(evidence.shape)))
                samples = np.unravel_index(rvs-1, evidence.shape)
                # raise e
            # print(samples)
            xs = [room[0, 0] + state.resolution[0] * row
                  for row in samples[0]]
            zs = [room[2, 0] + state.resolution[2] * col
                  for col in samples[1]]
            for x, z in zip(xs, zs):
                # print("x,z: %s, %s" % (x, z))
                if room[0, 0] <= x <= room[0, 1] \
                   and room[2, 0] <= z <= room[2, 1]:
                    # print("accepting...")
                    not_found = False
                    break
                print("not accepting %f, %f...room: %s" % (x, z, room.T))
        # print("Final x,z: %s, %s" % (x, z))
        # pick random object from category "label"
        obj_new = copy.deepcopy(np.random.choice(scene_objects[label]))
        assert obj_new._mesh._hull is not None, "why none?"
        # tmp = copy.deepcopy(obj_new)
        # for o in scene_objects[label]:
            # print("comparing %s with %s" % (o.name, obj_new.name))
            # assert obj_new is not o, "Need a clone!"
            # assert obj_new._mesh is not o._mesh, "Need a clone mesh!"
        # print("obj_new: %s" % obj_new)

        centroid = obj_new.transform[:3, 3]  # obj_new.get_centroid()
        angle = np.random.uniform(-np.pi, np.pi)
        tr = rot_y(angle)
        tr[0, 3] = x - centroid[0]
        tr[2, 3] = z - centroid[2]
        obj_new.apply_transform(tr)
        # if room[0, 0] > obj_new.transform[0, 3] \
        #    or room[0, 1] < obj_new.transform[0, 3] \
        #    or room[2, 0] > obj_new.transform[2, 3] \
        #    or room[2, 1] < obj_new.transform[2, 3]:
        #     vis = Visualizer()
        #     vis.add_coords()
        #     print("centroid: %s" % centroid.T)
        #     print("NOOO: %s\n(room:\n%s)" % (obj_new.transform[:3, 3], room))
        #     print("tr:\n%s" % tr)
        #     print("x,z: %f, %f" % (x, z))
        #     vis.add_mesh(tmp._mesh, "mesh0")
        #     vis.show()
        #     assert False
        # print("obj_new angle: %f vs orig: %f" %
        #       (obj_new.get_angles_euler(which=1), angle))
        # obj_new.apply_transform(
        #     translation([x - centroid[0], 0., z - centroid[2]]))
        # print("obj_new: %s" % obj_new.label)
        # with Timer('isec0', verbose=True) as t:
        #     intersects = False
        #     for ob in scene.objects.values():
        #         if ob.intersects(obj_new):
        #             intersects = True
        #             break
        # with Timer('isec1', verbose=True) as t:
        intersects = \
            next((ob for ob in scene.objects.values()
                  if ob.intersects(obj_new)), None) \
            is not None
        # i2_bool = i2 is not None
        # assert i2_bool == intersects, "No match: i2_bool(i2:%s):%s intersects:%s" % (i2, i2_bool, intersects)
        if not intersects:
            scene.add_object(-1, obj_new, clone=False)
            _labels.remove(label)

        # plt.figure()
        # plt.scatter(x, z)
        # plt.xlim(room[0, 0], room[0, 1] - state.resolution[0])
        # plt.ylim(room[2, 0], room[2, 1] - state.resolution[1])
        # plt.title(label)
        # plt.draw()
        # plt.show()
        # shape = evidence.shape
        # print("shape: %s" % repr(shape))
        # iteration2 = 0
        # prob_best = -1.
        # loc_best = (-1, -1)
        # while iteration2 < 10:
        #     x = np.randint(0, shape[0]+1)
        #     z = np.randint(0, shape[1]+1)
        #     try:
        #         obj_new = np.random.choice(
        #         if prob_best < evidence[x, z]:
        #             prob_best = evidence[x, z]
        #         loc_best = (x, z)
        #         iteration2 += 1
        #         _labels.remove(label)
        #         # iter = 0
        #         # while iter < 10:
    # plt.show()
    plt.close()
    return scene


def score(scene, dbn, gmms, ignore, show=False):
    """Scores a rendered scene
    :param ignore: labels to ignore (usually wall and floor)
    """
    raise DeprecationWarning("Deprecated")

    curr_labels = dict((k, [v])
                       for k, v in scene.get_labels(ignore=ignore).items())
    try:
        curr_score = dbn.score(curr_labels, True)
    except ValueError:
        print("Could not score with dbn, because of unseen cardinality..."
              "setting score to 0")
        curr_score = 0.

    # Eval GMM-s
    gen = ((oid0, oid1)
           for oid0, oid1 in product(scene.objects.keys(), scene.objects.keys())
           if scene.objects[oid0].label in gmms
           and scene.objects[oid1].label in gmms[scene.objects[oid0].label]
           and oid0 != oid1)
    if show:
        plt.figure()
        # plt.xlim(state.room[0, 0], state.room[0, 1] - state.resolution[0])
        # plt.ylim(state.room[2, 0], state.room[2, 1] - state.resolution[1])
    for oid0, oid1 in gen:
        # vis = Visualizer()
        # vis.add_coords()
        ob0 = scene.objects[oid0]
        ob1 = scene.objects[oid1]
        # c0 = ob0.get_centroid()
        # print("c0: %s" % c0)
        # print("tr.translation: %s" % ob0.transform[:3, 3])
        # pos = (ob1.get_centroid() - c0).squeeze()
        tr_inv = np.linalg.inv(ob0.transform)
        pos = np.matmul(tr_inv, ob1.transform[:4, 3])[:3]
        assert pos.shape == (3,), "wrong shape: %s" % repr(pos.shape)
        # print("pos: %s" % pos)
        gmm = gmms[ob0.label][ob1.label]

        # X, _ = gmm.estimator.sample(100)
        # print(X)
        # print("Y: %s" % Y)
        # plt.figure()
        # plt.title("%s-%s" % (ob0.label, ob1.label))
        # plt.scatter(X[:, 0], X[:, 1], c='k')
        c = ob0.transform[:3, 3]
        if show:
            plt.plot(c[0], c[2], "o", c='r')
        # plt.xlim(state.room[0, 0], state.room[0, 1] - state.resolution[0])
        # plt.ylim(state.room[2, 0], state.room[2, 1] - state.resolution[1])
        q = [[pos[0], pos[2]]]
        # print(dir(gmm))
        # print("means: %s" % gmm.estimator.means_)
        # print("means-pos: %s" % (gmm.estimator.means_ - pos[[0, 2]]))
        # plot_ellipses(
        #     plt.axes(), gmm.estimator.weights_,
        #     gmm.estimator.means_ + pos[[0, 2]],
        #     gmm.estimator.covariances_)
        # print("query: %s" % repr(q))
        _score = gmm.estimator.score(q)
        if show:
            plt.annotate("%f" % _score,
                         xy=(ob1.transform[0, 3], ob1.transform[2, 3]),
                         xytext=(c[0], c[2]),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         )
        # print("score %s-%s is %f" % (ob0.label, ob1.label, _score))
        # plt.show()
        # vis.add_mesh(ob0._mesh, "ob0")
        # vis.add_mesh(ob1._mesh, "ob1")
        # vis.show()
        curr_score += _score
    if show:
        plt.title("score")
        plt.show()
        plt.close()

    return curr_score