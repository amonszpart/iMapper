import argparse
import copy
import itertools
import os
import sys

# import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import shapely.geometry as geom
from descartes.patch import PolygonPatch
from scipy.interpolate import RegularGridInterpolator
from shapely.geos import TopologicalError
from shapely.ops import cascaded_union

from imapper.logic.mesh_OBJ import MeshOBJ
from imapper.logic.scenelet import get_scenelets
from imapper.util.stealth_logging import logging
try:
    from imapper.visualizer.visualizer import Visualizer
except ImportError:
    print("Could not import vtk")
from imapper.util.my_pickle import pickle

if not sys.version_info[0] < 3:
    from functools import lru_cache
else:
    from repoze.lru import lru_cache


def arc_shapely(center, radii, start_angle, end_angle, num_segs=6):
    assert len(radii) == 2, "Two radii expected: %s" % repr(radii)
    # The coordinates of the arc
    theta = np.linspace(start_angle, end_angle, num_segs)
    rev_theta = theta[::-1]
    x = np.concatenate((center[0] + radii[0] * np.cos(theta),
                        center[0] + radii[1] * np.cos(rev_theta)))
    y = np.concatenate((center[1] + radii[0] * np.sin(theta),
                        center[1] + radii[1] * np.sin(rev_theta)))
    # print(x.shape)
    # print(y.shape)

    arc = geom.Polygon(zip(x, y))
    # print("arc:\n%s" % arc)
    return arc


def radial_profile(data, center):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class Histogram(object):
    def __init__(self):
        super(Histogram, self).__init__()
        self._volume = dict()

    def get_pos_2d(self, x0, x1):
        raise NotImplementedError("This is an abstract class")

    def set_bin(self, label, x0, x1, value, polys=None):
        raise NotImplementedError("This is an abstract class")


class SquareHistogram(Histogram):
    def __init__(self, shape, edges):
        super(SquareHistogram, self).__init__()

        self._volume = dict()
        """Internal storage"""
        self._samplers = dict()
        """GridSamplers of volume"""
        self.shape = shape
        """Shape of histogram"""
        self.edges = [np.asarray(dim_edges) for dim_edges in edges]
        """Real world description of histogram cell borders"""

    def get_pos_2d(self, x0, x1):
        return [np.mean(self.edges[0, x0:x0+2]),
                np.mean(self.edges[1, x1:x1+2])]

    def set_bin(self, label, x0, x1, value, polys=None):
        raise RuntimeError("Should not be used because of grid ipol...")

        # try:
        #     self._volume[label][x0, x1] = value
        # except KeyError:
        #     self.volume[label] = \
        #         np.zeros(shape=self.shape, dtype='f4')
        #     self.volume[label][x0, x1] = value
        #
        # if polys is not None:
        #     logging.error("IGNORING non-empty polys")

    @classmethod
    def from_mat(cls, params, bins, categories):
        # reverse angular dim because it comes in rev order
        h = SquareHistogram(
           shape=np.squeeze(params['shape'][0][0]) - 1,
           edges=[
               np.squeeze(params['x_edges'][0][0]).astype('f4'),
               np.squeeze(params['y_edges'][0][0]).astype('f4'),
               np.squeeze(params['orientation_edges'][0][0]).astype('f4')[::-1]
           ]
        )
        # logging.info("edges:\n%s" % h.edges)

        # grid points are middle points between edges
        points = list(
            (h.edges[i][1:] + h.edges[i][:-1]) / np.float32(2.)
            for i in range(len(h.edges))
        )
        # for wrap around, we copy the last layer to the front
        # and the first layer to the back

        # average spacing between edges
        steps = [np.mean(h.edges[i][1:] - h.edges[i][:-1])
                 for i in range(len(h.edges))]

        # use this to step one more at front and back
        # xy: pad with 0-s for linear falloff
        points[0] = \
            np.concatenate(([points[0][0] - steps[0]],
                            points[0],
                            [points[0][-1] + steps[0]]))
        points[1] = \
            np.concatenate(([points[1][0] - steps[1]],
                            points[1],
                            [points[1][-1] + steps[1]]))
        # angle: wrap around
        points[2] = np.concatenate(([points[2][0] - steps[2]],
                                    points[2],
                                    [points[2][-1] + steps[2]]))
        for d in range(len(h.edges)):
            h.edges[d] = np.concatenate((
                [h.edges[d][0] - steps[d]],
                h.edges[d],
                [h.edges[d][-1] + steps[d]]
            )).astype('f4')


        # fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=2,
        #                          sharex=True, sharey=True)
        # axes = axes.ravel()

        # for each category
        for id_category, category in enumerate(np.squeeze(categories)):
            # get label from input map
            label = category[0]
            assert len(label) > 1, "Wrong label? %s" % label
            # reverse last dimension to match edges
            bin_ = bins[id_category, :, :, ::-1]
            # prepend last and append first of 3rd (angular) dimension
            bin_ = np.concatenate((np.expand_dims(bin_[:, :, -1], axis=2),
                                   bin_[:, :, :],
                                   np.expand_dims(bin_[:, :, 0], axis=2)),
                                  axis=2)
            # debug copy
            # bin_tmp = np.copy(bin_)
            # theta_id = 0
            # logging.info("bin before:\n%s" % bin_[:, :, theta_id])

            # add rows at edges
            bin_ = np.pad(bin_, pad_width=((1, 1), (1, 1), (0, 0)),
                          mode='constant', constant_values=0)
            # bin_ = np.concatenate((
            #     np.expand_dims(
            #         bin_[0, :, :]
            #         + (bin_[0, :, :] - bin_[1, :, :])
            #         / (points[0][1] - points[0][2])
            #         * (points[0][0] - points[0][1]), axis=0),
            #     bin_,
            #     np.expand_dims(
            #         bin_[-1, :, :]
            #         + (bin_[-1, :, :] - bin_[-2, :, :])
            #         / (points[0][-2] - points[0][-3])
            #         * (points[0][-1] - points[0][-2]), axis=0)),
            #     axis=0)

            # add columns at edges
            # bin_ = np.concatenate((
            #     np.expand_dims(
            #         bin_[:, 0, :]
            #         + (bin_[:, 0, :] - bin_[:, 1, :])
            #         / (points[1][1] - points[1][2])
            #         * (points[1][0] - points[1][1]), axis=1),
            #     bin_,
            #     np.expand_dims(
            #         bin_[:, -1, :]
            #         + (bin_[:, -1, :] - bin_[:, -2, :])
            #         / (points[1][-2] - points[1][-3])
            #         * (points[1][-1] - points[1][-2]), axis=1)),
            #     axis=1)

            # fix diagonals
            # bin_[0, 0, :] = bin_[0, 1, :] + bin_[1, 0, :] - bin_[1, 1, :]
            # bin_[0, -1, :] = bin_[0, -2, :] + bin_[1, -1, :] - bin_[1, -2, :]
            # bin_[-1, -1, :] = bin_[-1, -2, :] + bin_[-2, -1, :] - bin_[-2, -2, :]
            # bin_[-1, 0, :] = bin_[-1, 1, :] + bin_[-2, 0, :] - bin_[-2, 1, :]

            # logging.info("bin after:\n%s" % bin_[:, :, theta_id])

            if False:
                vmin = np.min(bin_[:, :, theta_id])
                vmax = np.max(bin_[:, :, theta_id])
                axes[0].imshow(
                    bin_tmp[:, :, theta_id], aspect='equal',
                    interpolation='nearest', cmap='jet',
                    extent=(points[0][0], points[0][-1], points[1][0], points[1][-1]))
                    # vmin=vmin, vmax=vmax)
                axes[0].set_xlim(points[0][0], points[0][-1])
                axes[0].set_ylim(points[1][0], points[1][-1])
                axes[0].grid(False)
                axes[1].imshow(
                    bin_[:, :, theta_id], aspect='equal',
                    interpolation='nearest', cmap='jet',
                    extent=(points[0][0], points[0][-1], points[1][0], points[1][-1]))
                    # vmin=vmin, vmax=vmax)
                axes[1].grid(False)
                axes[1].set_xlim(points[0][0], points[0][-1])
                axes[1].set_ylim(points[1][0], points[1][-1])
                assert bin_.shape == \
                    (len(points[0]), len(points[1]), len(points[2])), \
                    "Wrong shape: %s vs. %s" \
                    % (bin_.shape, (len(points[0]), len(points[1]), len(points[2])))
                logging.info("bin shape: %s" % repr(bin_.shape))
                plt.suptitle(label)
                if label == u'table' or label == u'couch':
                    plt.show()
            # store volume
            h._volume[label] = bin_
            # create grid interpolator
            sampler = \
                RegularGridInterpolator(points=points,
                                        values=h._volume[label],
                                        bounds_error=True)
            assert all(
               (
                   sampler.grid[d][0] > h.edges[d][0] and
                   sampler.grid[d][-1] < h.edges[d][-1]
                   for d in range(len(h.edges))
               )
            ), "Edges inside grid:\n%s\n%s" % (
                tuple((sampler.grid[d][0], sampler.grid[d][-1])
                      for d in range(len(sampler.grid))),
                tuple((h.edges[d][0], h.edges[d][-1])
                      for d in range(len(h.edges)))
            )

            h._samplers[label] = sampler

            # test
            # qs = list((h.edges[i][1:] + h.edges[i][:-1]) / np.float32(2.)
            #           for i in range(len(h.edges)))
            # for x, y in product(range(len(h.edges[0])-1), range(len(h.edges[1])-1)):
            #     logging.info("x: %s, y: %s" % (x, y))
            #     logging.info("qs[0][x]: %s, qs[1][y]: %s, h.edges[2][0]: %s"
            #                  % (qs[0][x], qs[1][y], qs[2][0]))
            #     logging.info("volume: %s, sampler: %s"
            #                  % (
            #                      h._volume[label][x+1, y+1, 1],
            #                      h._samplers[label]((qs[0][x], qs[1][y], qs[2][0]))
            #                  )
            #                  )
        return h

    @lru_cache(maxsize=10)
    def get_grid_bounds(self, label):
        grid = self._samplers[label].grid
        return np.array(
           [
               [grid[d][0], grid[d][-1]]
               for d in range(len(grid))
           ]
        )

    def contains(self, p_local, label):
        bounds = self.get_grid_bounds(label)
        assert len(bounds) == 3, "no: %s" % repr(bounds.shape)
        try:
            return all(
               [
                   bounds[i][0] <= p_local[i] <= bounds[i][1]
                   for i in range(len(bounds))
               ]
            )
        except KeyError:
            return False

    def get_value(self, p_local, label):
        ndim = len(self.edges)
        assert (p_local.ndim == 1 and p_local.shape[0] == ndim) \
               or ndim == p_local.shape[1], \
            "Dimension mismatch: %s %s" % (ndim, p_local.shape)

        # throws KeyError, if we don't have label
        _slice = self._samplers[label]
        return _slice(p_local)

        # c2 = tuple(
        #     next(i
        #          for i in range(self.edges[d].size-1)
        #          if self.edges[d][i] <= p_local[d] < self.edges[d][i+1])
        #     for d in range(len(self.edges)))
        # if len(c2) != len(_slice.shape):
        #     raise ValueError("Does not contain")
        # return _slice[c2]

    def has_label(self, label):
        return label in self._samplers


class RadialHistogram(Histogram):
    _two_pi = np.float32(2. * np.pi)

    def __init__(self, shape, r_max=2., angular_offset=np.pi/4.):
        super(RadialHistogram, self).__init__()
        # self.sections = sections
        self.shape = shape
        # """How many angular bins"""
        # self.layers = layers
        # """How many radial layers"""
        self.r_max = r_max
        """Max radius of radial histogram"""

        self.angular_edges = \
            np.linspace(angular_offset,
                        RadialHistogram._two_pi + angular_offset,
                        self.shape[0]+1, endpoint=True,
                        dtype=np.float32)
        self.radial_edges = \
            np.linspace(0., r_max, self.shape[1]+1, endpoint=True,
                        dtype=np.float32)
        self.arc_polygons = \
            dict(
                (
                    (bin_angular, bin_radial),
                    arc_shapely((0., 0.), (start_radius, end_radius),
                                start_angle, end_angle)
                )
                for (bin_angular, (start_angle, end_angle)),
                    (bin_radial, (start_radius, end_radius))
                in itertools.product(
                    enumerate(
                        zip(self.angular_edges[:-1], self.angular_edges[1:])),
                    enumerate(
                        zip(self.radial_edges[:-1], self.radial_edges[1:]))
                )
            )
        """Represents the approximated arcs:
        (angular bin id, angular radial id) => shapely polygon"""

        assert isinstance(self.arc_polygons, dict), \
            "type(arc_polygons): %s" % type(self.arc_polygons)

        self.volume = {}
        """Stores a 2d bin array for each object category (chair, table, etc.)
        """
        self.polys_debug = {}

        self.poses = None
        """3D poses associated with histogram (n_frames x 3 x n_joints)"""
        self.descriptor = None
        """Descriptor of the poses"""
        self.transforms = []
        """Transforms to get the scenelet into its current state"""
        self.forward = None
        """Scenelet forward direction in world space at mid_frame +-2"""

    @classmethod
    def from_mat(cls, angular_edges, radial_edges, bins, categories):
        h = RadialHistogram(shape=(len(angular_edges)-1, len(radial_edges)-1),
                            r_max=radial_edges[-1],
                            angular_offset=angular_edges[0])
        for id_category, category in enumerate(categories):
            label = category[0][0]
            h.volume[label] = bins[id_category, :, :]
        assert np.allclose(h.angular_edges, np.squeeze(angular_edges.T)), \
            "No:\n%s\n%s" % (h.angular_edges, angular_edges)
        assert np.allclose(h.radial_edges, np.squeeze(radial_edges.T)), \
            "No:\n%s\n%s" % (h.radial_edges, radial_edges)
        return h

    def set_bin(self, label, x0, x1, value, polys=None):
        """

        :param label:
        :param x0: bin_angular
        :param x1: bin_radial
        :param value:
        :param polys:
        :return:
        """
        try:
            self.volume[label][x0, x1] = value
        except KeyError:
            self.volume[label] = \
                np.zeros(shape=self.shape, dtype='f4')
            self.volume[label][x0, x1] = value

        if polys is not None:
            try:
                self.polys_debug[label].extend(polys)
            except KeyError:
                self.polys_debug[label] = polys

    def get_pos_2d(self, x0, x1):
        """
        Converts bin ids to 2d positions
        :param x0: bin_angular
        :param x1: bin_radial
        """
        angle = np.mean(self.angular_edges[x0:x0+2])
        radius = np.mean(self.radial_edges[x1:x1+2])
        return pol2cart(radius, angle)

    def plot(self, show=False, polys_show=None):
        """
        :param show:
        :param polys_show: {label: [(poly, weight), ...]}
        :return:
        """
        nrows = int(np.ceil(np.sqrt(len(self.volume))))
        fig, axes = plt.subplots(nrows, nrows, sharey=True, sharex=True)
        axes = axes.ravel()
        for ax_id, (label, slice) in enumerate(self.volume.items()):
            ax = axes[ax_id] if len(self.volume) > 1 else axes
            hits = np.argwhere(slice > 0.)
            for elem in hits:
                # angle = np.mean(self.angular_edges[elem[0]:elem[0]+2])
                # radius = np.mean(self.radial_edges[elem[1]:elem[1]+2])
                # pos = pol2cart(radius, angle)
                # pos = self.get_pos_2d(elem[0], elem[1])
                # ax.text(pos[0], pos[1], "%.2f, %.2f" % (np.rad2deg(angle), radius))
                poly_arc = self.arc_polygons[(elem[0], elem[1])]
                patch = PolygonPatch(poly_arc, facecolor='b',
                                     edgecolor='r', alpha=0.3, zorder=2)
                ax.add_patch(patch)

            seen = []
            if label in self.polys_debug:
                for bin_angular, bin_radial, poly, perc in self.polys_debug[label]:
                    pos = self.get_pos_2d(bin_angular, bin_radial)
                    ax.text(pos[0]-0.05, pos[1], "%.1f%%" % (perc * 100.))
                    ax.add_patch(
                        PolygonPatch(poly, facecolor='r',
                                     edgecolor='g', alpha=0.8))
                    seen.append((bin_angular, bin_radial))
            ax.set_title(label)

            if polys_show is not None and label in polys_show:
                for poly, weight in polys_show[label]:
                    ax.add_patch(
                        PolygonPatch(poly, facecolor='r',
                                     edgecolor='g', alpha=0.8))
                    pos = poly.centroid
                    print(dir(pos))
                    ax.text(pos.x-0.05, pos.y, "%f" % weight)
                    print("added poly %s at %f, %f" % (label, pos.x, pos.y))

            for (bin_angular, bin_radial), poly_arc in self.arc_polygons.items():
                if (bin_angular, bin_radial) not in seen:
                    ax.add_patch(PolygonPatch(poly_arc, facecolor='b',
                                              edgecolor='r', alpha=0.1))
                    pos = self.get_pos_2d(bin_angular, bin_radial)
                    ax.text(pos[0]-0.05, pos[1],
                            "%.1f%%" %
                            (self.volume[label][bin_angular, bin_radial] * 100.))
                else:
                    print("hm %s?" % repr((bin_angular, bin_radial)))
            #     ax.add_patch(patch)
            ax.set_xlim(-self.radial_edges[-1], self.radial_edges[-1])
            ax.set_ylim(-self.radial_edges[-1], self.radial_edges[-1])
            ax.set_aspect('equal')
        if show:
            plt.show()

    def get_weight(self, label, poly):
        """Returns the weight of the object based on area overlap
        with this weighted histogram
        charness = \frac{1.0}{\sum w_{area}}
                   \sum \left(w_{bin}
                            \frac{A_{intersection}}{min(A_{arc}, A_{object})}
                        \right)
        """
        if label not in self.volume:
            logging.info("Don't have %s charness" % label)
            return 0.

        # w_sum = 0.
        poly_area = poly.area
        sum_area = 0.
        charness = 0.
        for (bin_angular, bin_radial), poly_arc in self.arc_polygons.items():
            poly_arc_area = poly_arc.area
            sum_area += poly_arc_area
            intersection = None
            try:
                intersection = poly.intersection(poly_arc)
            except TopologicalError:
                logging.error("Topology error, skipping")
                continue
            if intersection is not None:
                # weight = intersection.area / min(poly_area, poly_arc_area)
                weight = intersection.area
                # logging.info(list(self.volume.keys()))
                try:
                    charness += \
                        weight \
                        * self.volume[str(label)][bin_angular, bin_radial]
                    # w_sum += weight
                except KeyError:
                    logging.info("Don't have %s in charness" % label)

        # return charness / w_sum if w_sum > 0. else charness
        return charness / min(poly_area, sum_area)

    def to_mdict(self, name):
        """Creates a dictionary ready to be saved to .mat files"""
        return {
            "name": name.split("::"),
            "sections": self.sections,
            "layers": self.layers,
            "r_max": self.r_max,
            "angular_edges": self.angular_edges,
            "radial_edges": self.radial_edges,
            "hists": dict((str(k), v) for k, v in self.volume.items()),
            "poses": self.poses,
            "descriptor": self.descriptor,
            "transforms": self.transforms,
            "forward": self.forward
        }


def characteristic_scenelets(py_scenes,
                             min_coverage=0.01, min_object_coverage=0.5,
                             sections=4, layers=2, r_max=1.5,
                             ignore={u'floor', u'wall'}):
    """
    :param py_scenes: 
    :param min_coverage: Percentage of histogram bin area covered by an object
    :param min_object_coverage: Percentage of object area covered by 
                                histogram bin
    :param sections: angular sections in histogram
    :param layers: radial layers in histogram
    :param r_max: max histogram radius
    :param ignore: object categories to ignore
    :return: 
    """
    np.set_printoptions(suppress=True)

    # sclt_angles = np.asarray(sclt_angles, dtype=np.float32)
    # sclt_angles = np.concatenate(sclt_angles).astype(np.float32)

    unique_labels = set(str(obj.label)
                        for scene in py_scenes.values()
                        for scenelet in scene.values()
                        for obj in scenelet.objects.values()
                        if obj.label not in ignore)

    print("unique labels: %s" % unique_labels)

    # output collections
    hists = []
    matlab_dicts = []

    # prototype histogram
    h_orig = Histogram(sections, layers, r_max)

    # for each scenelet
    for name_scene, name_scenelet, scenelet in get_scenelets(py_scenes):
        # name_scene = names[0]
        # name_scenelet = names[1]

        # copy prototype histogram to output
        hists.append(copy.deepcopy(h_orig))
        # reference it
        h = hists[-1]

        # ensure all bins appear in the histogram by setting bin 0,0 to 0.
        for ulabel in unique_labels:
            h.set_bin(str(ulabel), 0, 0, 0.)

        # reference skeleton in scenelet
        skel = scenelet.skeleton
        # sorted frames of skeleton in scenelet
        skel_frames = skel.get_frames()
        # middle frame of scenelet
        frame_id = skel_frames[len(skel_frames)//2]
        # get pose at middle time in scenelet
        pose = skel.get_pose(frame_id)

        # copy skeleton to histogram
        h.poses = skel._poses

        # copy angles (descriptor) to histogram
        # returns frame_ids as well, hence [0]
        h.descriptor = skel.get_angles()[0]
        # copy scenelet transforms
        h.transforms = scenelet.aux_info[u'preproc'][u'transforms']
        # copy scenelet forward direction
        h.forward = scenelet.aux_info[u'forward']

        # debug
        # print("%s\n.\n%s =\n%s" % (
        #     np.asarray(h.transforms[1])[:3, :3],
        #     np.asarray(h.forward).T,
        #     np.dot(np.asarray(h.transforms[1])[:3, :3],
        #     np.asarray(h.forward).T)))

        # sort object part bounding rectangles per object category
        # polys = \
        #     dict((obj.label,  # key
        #           cascaded_union(  # merged top-view projected obb-s
        #               [
        #                   geom.Polygon(
        #                       part.obb.corners_3d_lower(
        #                           up_axis=(0., -1., 0.))[:, [0, 2]]
        #                   )
        #                   for part in obj.parts.values()
        #               ]
        #           ))
        #          for obj in scenelet.objects.values()
        #          if obj.label not in ignore)

        # same without list comprehensions:
        # gather merged top-view polygons per label
        polys = dict()
        # for each object if not floor or wall
        for obj in (obj for obj in scenelet.objects.values()
                    if obj.label not in ignore):

            # collect object part polygons
            locals = []
            for part in obj.parts.values():
                locals.append(geom.Polygon(
                    part.obb.corners_3d_lower(
                        up_axis=(0., -1., 0.))[:, [0, 2]]
                ))

            # if has at least one part
            if len(locals):
                # add already stored polygon under this category, if exists
                try:
                    locals.insert(0, polys[obj.label])
                except KeyError:
                    pass

                # try taking the union of the parts of the object
                try:
                    # add to the per-category polygon dictionary
                    polys[obj.label] = cascaded_union(locals)
                except ValueError:
                    # debug cascaded union
                    plt.figure()
                    ax = plt.gca()
                    for pl in locals:
                        patch = PolygonPatch(
                            pl, facecolor='b',
                            edgecolor='r', alpha=0.5, zorder=2)
                        ax.add_patch(patch)
                    plt.show()

        # For each segment in the histogram
        for (bin_angular, bin_radial), arc_poly in h.arc_polygons.items():
            # estimate area threshold TODO: scale by increasing bin_radial
            thresh_area_arc_poly = arc_poly.area * min_coverage
            # for each category assembled above
            for label, poly in polys.items():

                percentage_intersection = 0.
                try:
                    intersection = poly.intersection(arc_poly)
                except TopologicalError:
                    print("topology error, skipping")
                    continue
                if intersection.is_empty:
                    is_covering = \
                        poly.contains(arc_poly) \
                        or poly.covers(arc_poly) \
                        or arc_poly.within(poly)
                if not intersection.is_empty:
                    is_intersecting = False
                    area_intersection = intersection.area
                    if area_intersection > thresh_area_arc_poly:
                        is_intersecting = True
                        percentage_intersection = \
                            area_intersection / arc_poly.area
                    else:
                        area_poly = poly.area
                        if area_intersection / area_poly > min_object_coverage:
                            is_intersecting = True
                else:
                    is_intersecting = False

                if is_intersecting or is_covering:
                    h.set_bin(
                        label, bin_angular, bin_radial,
                        percentage_intersection,
                        polys=[(bin_angular, bin_radial, poly,
                                percentage_intersection)])
                else:
                    h.set_bin(label, bin_angular, bin_radial, 0.)

        # copy to MATLAB output
        matlab_dicts.append(h.to_mdict("%s::%s" % (name_scene, name_scenelet)))

    # save MATLAB output
    scipy.io.savemat("hists.mat", {'histograms': matlab_dicts})
    print("Saved to %s" % os.path.abspath("hists.mat"))

    # Show last scene in 3D
    with Visualizer() as vis:
        vis.add_coords()

        for oid, obj in scenelet.objects.items():
            if obj.label not in ignore:
                for part in obj.parts.values():
                    vis.add_mesh(MeshOBJ.from_obb(part.obb),
                                 "%02d_%s_%s" % (oid, obj.label, part.label))

        for jid in range(pose.shape[1]):
            vis.add_sphere(pose[:, jid], 0.05, (0.1, 0.9, 0.9),
                           "joint_%d" % jid)


        camera = vis._ren.GetActiveCamera()
        camera.SetViewUp(0., -1., 0.)
        vis._ren.ResetCamera()
        vis.show()


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Radial histogram generation")
    parser.add_argument(
        "-d", dest="path_scenelets",
        default="/media/Data1/ResearchData/stealth/data-pigraphs/" \
                "stealth_scenelets_subsample010_framestep015_gap-8.real")
    args = parser.parse_args()

    path_scenelets = args.path_scenelets

    # read scenelets
    _path_scenes_pickle = os.path.join(path_scenelets, 'scenes.pickle')
    if os.path.exists(_path_scenes_pickle):
        print("Reading scenelets from pickle: %s, delete this file, "
              "if you want to reparse the scenelets" % _path_scenes_pickle)
        py_scenes = pickle.load(open(_path_scenes_pickle, 'rb'))
    else:
        print("Listing \"%s\":\n" % path_scenelets)
        py_scenes = {}
        for parent, dirs, files in os.walk(path_scenelets):
            scene_name = None
            for f in [f for f in files
                      if f.endswith('.json') and f.startswith('skel')]:
                if not scene_name:
                    scene_name = os.path.basename(os.path.split(parent)[-1])
                    py_scenes[scene_name] = {}
                j_path = os.path.join(parent, f)
                # print("j_path: %s" % j_path)
                scenelet = Scenelet.load(j_path)
                if len(scenelet.objects):
                    py_scenes[scene_name][os.path.splitext(f)[0]] = scenelet

        pickle.dump(py_scenes, open(_path_scenes_pickle, 'wb'))
        print("Dumped py_scenes to %s" % _path_scenes_pickle)

    characteristic_scenelets(py_scenes)

# characteristic_scenelets(sclt_angles, rev_keys, py_scenes)
# sys.exit(0)

    # import matplotlib
    # print("backend: %s" % matplotlib.get_backend())
    # # matplotlib.use('Qt5Agg')
    # # from PyQt5 import QtGui
    # import matplotlib.pyplot as plt
    # from descartes.patch import PolygonPatch
    #
    # h = Histogram()
    #
    # f = plt.figure()
    # ax = f.gca()
    # for polygon in h.arc_polygons:
    #     plot_coords(ax, polygon.exterior)
    #     patch = PolygonPatch(polygon, facecolor='b',
    #                          edgecolor='r', alpha=0.5, zorder=2)
    #     ax.add_patch(patch)
    #
    # # polygon = arc_shapely((1., 1.), [1., 2.], 0., np.pi/2.)
    # # print(polygon.area)
    # # plot_coords(ax, polygon.exterior)
    # # patch = PolygonPatch(polygon, facecolor='b',
    # #                      edgecolor='r', alpha=0.5, zorder=2)
    # # ax.add_patch(patch)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.show()

