from imapper.visualization.plotting import plt

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import shapely.geometry as geom
import shapely.affinity as saffinity
from shapely.ops import cascaded_union

from imapper.logic.mesh_OBJ import MeshOBJ, Scene
from imapper.logic.scenelet import Scenelet
from imapper.logic.state import State
from imapper.scenelet_fit.consts import TWO_PI
from imapper.util.my_pickle import pickle, pickle_load
from imapper.util.stealth_logging import lg
from imapper.logic.categories import COLORS_CATEGORIES, \
    TRANSLATIONS_CATEGORIES, CATEGORIES_DOMINANT, CATEGORIES
from descartes.patch import PolygonPatch

try:
    from stealth.visualizer.visualizer import Visualizer
    no_vis = False
except ImportError as e:
    print("Vtk error %s" % e)
    no_vis = True

# export PYTHONPATH=$(pwd)/stealth:PYTHONPATH; \
# python stealth/scenelet_fit/create_dataset.py /data/data/shared/3bHallway-mati2-2014-04-30-22-42-17 //NOLINT
_TWO_PI = np.pi * 2.
_HALF_PI = np.pi / 2.


def get_rectangle(poly, angle):
    """

    :param poly:
    :param angle:
    :return:
        [cx, cy, theta, sx, sy]
    """
    poly2 = saffinity.affine_transform(
        poly,
        [np.cos(angle), -np.sin(angle),
         np.sin(angle), np.cos(angle),
         0, 0]  # a, b, d, e, cx, cy
    )

    params = [poly.centroid.x, poly.centroid.y, angle,
              poly2.bounds[2] - poly2.bounds[0],
              poly2.bounds[3] - poly2.bounds[1]]

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_artist(PolygonPatch(poly,
                                   facecolor='b',
                                   alpha=0.9))
        lg.debug("poly: %s,\nangle: %s" % (poly, np.rad2deg(angle)))
        lg.debug("bounds: %s" % repr(poly.bounds))
        lg.debug("poly2.centroid: %s" % poly2.centroid)
        lg.debug("poly.centroid: %s" % poly.centroid)
        dc = (poly.centroid.x - poly2.centroid.x, poly.centroid.y - poly2.centroid.y)
        lg.debug("dc: %s" % repr(dc))
        poly2 = saffinity.translate(poly2, dc[0], dc[1])
        ax.add_artist(PolygonPatch(
            poly2, facecolor='r', alpha=0.8
        ))
        ax.set_xlim(min(poly.bounds[0], poly2.bounds[0]),
                    max(poly.bounds[2], poly2.bounds[2]))
        ax.set_ylim(min(poly.bounds[1], poly2.bounds[1]),
                    max(poly.bounds[3], poly2.bounds[3]))

        plt.show()

    return params


def get_poly(obbs):
    poly = cascaded_union(
       [
           geom.Polygon(obb.corners_3d_lower(
              up_axis=(0., -1., 0.))[:, [0, 2]])
           for obb in obbs
       ])
    return poly


class GridPoly(object):
    def __init__(self, poly, area, xy, occupancy=0.):
        self.poly = poly
        self.area = area
        self.occupancy = occupancy
        self.xy = xy


def get_grid_shapely(occup, res_orig):
    assert res_orig.shape == (4, ), "Expected xyz-theta resolution"
    room = occup.room
    assert room.shape == (3, 2), "Assumed 3D room shape"
    res_target = (occup.resolution[0], occup.resolution[2])

    start_x = room[0, 0] - res_orig[0] / 2.
    end_x = room[0, 1] + res_orig[0] / 2.
    span_x = (end_x - start_x)
    diff_x = span_x - int(span_x / res_target[0]) * res_target[0]
    start_x += diff_x / 2.
    end_x -= res_target[0]  # last square ends at end_x + res_target

    start_y = room[2, 0] - res_orig[2] / 2.
    end_y = room[2, 1] + res_orig[2] / 2.
    span_y = (end_y - start_y)
    diff_y = span_y - int(span_y / res_target[1]) * res_target[1]
    start_y += diff_y / 2.
    end_y -= res_target[1]  # last square ends at end_y + res_target

    grid_size = occup.get_grid_size()
    polys = []
    lg.debug("start_x: %s, end_x: %s, res_target: %s" % (start_x, end_x, res_target))
    for ix, x in enumerate(np.arange(start_x, end_x, res_target[0])):
        if ix >= grid_size[0]:
            continue
        for iy, y in enumerate(np.arange(start_y, end_y, res_target[1])):
            if iy >= grid_size[2]:
                continue
            box = geom.box(x, y, x + res_target[0], y + res_target[1])
            polys.append(GridPoly(box, area=box.area, xy=(ix, iy)))
        #assert iy == grid_size[2] - 1, "No: %s %s" % (iy, grid_size)
    #assert ix == grid_size[0] - 1, "No: %s %s %s" % (ix, x, grid_size)
    return polys


def main(argv=None):
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
       'd', help="Folder of scene"
    )
    parser.add_argument(
       '-resolution', help='Target resolution for occupancy map', default=0.1
    )
    parser.add_argument(
       '-thresh-area',
       help='Ratio of occupancy map cell area that has to be occupied '
            'for it to count as occupied',
       default=0.1
    )
    parser.add_argument(
       '-postfix', type=str, help="Scene postfix for augmentation", default=""
    )
    args = parser.parse_args(argv if argv is not None else sys.argv)
    res_target = args.resolution
    if args.postfix and len(args.postfix) and not args.postfix.startswith('_'):
        args.postfix = "_%s" % args.postfix

    path_parent, name_input = os.path.split(os.path.abspath(args.d))
    lg.warning("name input: %s" % name_input)
    path_for_tf = os.path.abspath(os.path.join(path_parent, os.pardir, 'dataset'))
        # if 'video' not in path_parent else os.path.join(path_parent, 'dataset')
    if not os.path.exists(path_for_tf):
        os.makedirs(path_for_tf, mode=0o0775)

    lg.debug("Loading scenelet...")
    path_scenelet = os.path.join(args.d, "skel_%s.json" % name_input)
    scenelet = Scenelet.load(path_scenelet)
    lg.debug("Scenelet: %s" % scenelet)

    path_state_pickle = os.path.join(args.d, "state%s.pickle" % args.postfix)
    if not os.path.exists(path_state_pickle):
        lg.error("Does not exist: %s" % path_state_pickle)
        return False

    # assert os.path.exists(path_state_pickle), \
    #     "Does not exist: %s" % path_state_pickle
    lg.debug("Loading volume...")
    state = pickle_load(open(path_state_pickle, 'rb'))
    lg.debug("Loaded volume...")

    lg.debug("Creating scene from scenelet")
    if not no_vis:
        vis = Visualizer(win_size=(1024, 1024))
        vis.add_coords()
    else:
        vis = None
    # scene = Scene(scenelet.name_scenelet)
    # colors = {0: (200., 0., 0.), 1: (0., 200., 0.), 2: (0., 0., 200.)}
    # unit_x = np.array((1., 0., 0.))

    occup = State(
       room=state.room, tr_ground_inv=None, res_theta=state.resolution[3],
       resolution=[res_target, res_target, res_target])
    occup.get_volume(labels_to_lin_ids_arg=state.get_labels_to_lin_ids())
    occup_angle = np.ones(
       shape=(
           len(occup.volume),
           occup.volume[0].shape[0],
           occup.volume[0].shape[1],
           1),
       dtype=np.float32
    ) * -1.
    assert np.min(occup_angle) < 0. and np.max(occup_angle) < 0., "Not empty"

    grid_polys = get_grid_shapely(occup=occup, res_orig=state.resolution)
    occup.volume.flags.writeable = True
    volume_occp = occup.volume
    angles = sorted(state.get_angles())
    labels_to_lin_ids = occup.get_labels_to_lin_ids()
    had_vtk_problem = no_vis

    plt.figure()

    rects = []
    for oid, ob in scenelet.objects.items():
        assert oid >= 0, "Need positive here"
        label = ob.label
        if label in TRANSLATIONS_CATEGORIES:
            label = TRANSLATIONS_CATEGORIES[label]

        if label not in labels_to_lin_ids:
            continue

        try:
            poly = get_poly([part.obb for part in ob.parts.values()])
        except ValueError as e:
            print("\n===========\n\nShapely error: %s for %s\n\n"
                  % (e, (label, oid, ob)))
            with open('error.log', 'a') as f:
                f.write("[%s] %d, %s, %s\n" % (args.d, oid, label, ob))
            continue

        ob_angle = ob.get_angle(positive_only=True)
        assert 0. <= ob_angle <= 2 * np.pi, "No: %g" % ob_angle

        rect = get_rectangle(poly, ob_angle)
        rect.extend([oid, CATEGORIES[label]])
        rects.append(rect)

        cat_id = labels_to_lin_ids[label] # cat_id in volume, not categories
        for gp in grid_polys:
            # skip, if not occupied enough
            if gp.poly.intersection(poly).area / gp.area < args.thresh_area:
                continue
            # save occupancy
            gp.occupancy = 1.
            id_angle_lower = None
            id_angle_upper = None
            if ob_angle > angles[-1]:
                id_angle_lower = len(angles) - 1
                id_angle_upper = 0
            else:
                for id_angle, angle in enumerate(angles):
                    if ob_angle < angle:
                        id_angle_upper = id_angle
                        id_angle_lower = id_angle - 1
                        break
            assert id_angle_lower is not None \
                   and id_angle_upper is not None, \
                "Wrong?"
            assert id_angle_upper != id_angle_lower, \
                "? %s %s" % (id_angle_lower, id_angle_upper)

            # cache
            xy = gp.xy

            # zero means empty in occupancy,
            # so object ids are shifted with 1
            # we need object ids to filter "untouched" objects
            # in tfrecords_create
            if volume_occp[cat_id, xy[0], xy[1], id_angle_lower] == 0 \
               or label in CATEGORIES_DOMINANT:
                volume_occp[cat_id, xy[0], xy[1], id_angle_lower] = oid + 1
            if volume_occp[cat_id, xy[0], xy[1], id_angle_upper] == 0 \
               or label in CATEGORIES_DOMINANT:
                volume_occp[cat_id, xy[0], xy[1], id_angle_upper] = oid + 1

            # angles are right now not per-category, but per-scene
            # hence, an object can only overwrite, if it's usually "above"
            # other objects, e.g. a table
            # this is a hack for a z-test
            if occup_angle[cat_id, xy[0], xy[1], 0] < 0. \
               or label in CATEGORIES_DOMINANT:
                occup_angle[cat_id, xy[0], xy[1], 0] = ob_angle

        if not had_vtk_problem:
            color = COLORS_CATEGORIES[label] if label in COLORS_CATEGORIES \
                else (200., 200., 200.)
            try:
                for id_part, part in ob.parts.items():
                    vis.add_mesh(MeshOBJ.from_obb(part.obb),
                                 name="ob_%02d_part_%02d" % (oid, id_part),
                                 color=color)
            except AttributeError:
                print("VTK problem...")
                had_vtk_problem = True
    #plt.savefig()
    plt.close()
    if not had_vtk_problem:
        vis.set_camera_pos(pos=(0., -1., 0.))
        vis.camera().SetFocalPoint(0., 0., 0.)
        vis.camera().SetViewUp(-1., 0., 0.)
        vis.set_camera_type(is_ortho=True)
        vis.camera().SetParallelScale(3.)
    # vis.show()

    name_recording = "%s_%s" % (os.path.basename(args.d), args.postfix) \
        if args.postfix else os.path.basename(args.d)
    lg.info("name_recording: %s" % name_recording)

    path_out_occp = os.path.join(
       os.path.dirname(args.d), os.pardir, 'occupancy', name_recording)
    if not os.path.exists(path_out_occp):
        os.makedirs(path_out_occp)

    # prepare www storage
    www_grid = {'evidence': {}, 'occ': {}}

    # normalize evidence maps
    vmax = 0.
    ims = {}
    for cat, cat_id in labels_to_lin_ids.items():
        ims[cat] = np.squeeze(
           np.sum(state.volume[cat_id, :, :, :], axis=2,
                  keepdims=True))
        vmax = max(vmax, np.max(ims[cat]))

    # gather joined occupancy map
    im_sum = None

    # for each evidence category
    for cat, cat_id in labels_to_lin_ids.items():
        im = ims[cat] / vmax * 255.
        path_out_im = os.path.join(path_out_occp, "e_%s.jpg" % cat)
        cv2.imwrite(path_out_im, im)
        # lg.debug("wrote to %s" % path_out_im)
        www_grid['evidence'][cat] = path_out_im

        im = np.squeeze(volume_occp[cat_id, :, :, 0])
        path_out_im = os.path.join(path_out_occp, "o_%s.jpg" % cat)
        cv2.imwrite(path_out_im, im * 255.)
        # lg.debug("wrote to %s" % path_out_im)
        www_grid['occ'][cat] = path_out_im

        if im_sum is None:
            im_sum = im.copy()
        else:
            im_sum = np.maximum(im, im_sum)

    #
    # save dataset
    #
    name_input_old = name_input
    if args.postfix is not None and len(args.postfix):
        name_input = "%s_%s" % (name_input, args.postfix)

    # state
    path_state_dest = os.path.join(path_for_tf, "state_%s.pickle" % name_input)
    shutil.copyfile(path_state_pickle, path_state_dest)
    lg.info("Copied\n\t%s to\n\t%s" % (path_state_pickle, path_state_dest))

    # occupancy
    path_occup_dest = os.path.join(path_for_tf, "occup_%s.pickle" % name_input)
    pickle.dump(occup, open(path_occup_dest, 'wb'), -1)
    lg.info("Wrote to %s" % path_occup_dest)

    # occupancy_angle
    path_occup_angle_dest = os.path.join(
       path_for_tf, "angle_%s.npy" % name_input)
    min_angle = np.min(occup_angle)
    assert min_angle < 0., "No empty cells??"
    lg.debug("min angle is %s" % min_angle)
    np.save(open(path_occup_angle_dest, 'wb'), occup_angle)
    lg.info("Wrote to %s" % path_occup_angle_dest)

    # skeleton
    path_copied = shutil.copy2(path_scenelet, path_for_tf)
    lg.info("Copied\n\t%s to \n\t%s" % (path_scenelet, path_copied))

    # charness skeleton
    name_skeleton_charness = "skel_%s-charness.json" % name_input_old
    path_scenelet_charness = os.path.join(args.d, name_skeleton_charness)
    assert os.path.exists(path_scenelet_charness), \
        "Does not exist: %s" % path_scenelet_charness
    shutil.copy2(path_scenelet_charness, path_for_tf)
    assert os.path.exists(os.path.join(path_for_tf, name_skeleton_charness)), \
        "Does not exist: %s" % os.path.join(path_for_tf,
                                            name_skeleton_charness)

    # rectangles
    name_rectangles = "rectangles_%s.npy" % name_input_old
    path_rectangles = os.path.join(path_for_tf, name_rectangles)
    np.save(open(path_rectangles, 'wb'), rects)

    #
    # visualize
    #

    path_out_im = os.path.join(path_out_occp, '3d.png')
    if not had_vtk_problem:
        vis.save_png(path_out_im)
    www_grid['3d'] = path_out_im

    path_out_im = os.path.join(path_out_occp, 'o_sum.png')
    max_im_sum = np.max(im_sum)
    if max_im_sum > 0.:
        cv2.imwrite(path_out_im, im_sum / max_im_sum * 255.)
    else:
        cv2.imwrite(path_out_im, im_sum * 255.)
    www_grid['o_sum'] = path_out_im

    path_www = os.path.join(path_out_occp, os.pardir)
    with open(os.path.join(path_www, 'index.html'), 'a') as f:
        f.write("<style> img {image-rendering: pixelated; } </style>\n")
        f.write("<script>\n")
        f.write("</script>\n")
        f.write("<h3>%s</h3>" % os.path.basename(args.d))
        f.write('<table>\n')

        f.write("<tr>\n")
        f.write("<th>3d</th>")
        f.write("<th>Occupancy sum</th>")
        for cat in www_grid['evidence']:
            f.write("\t<th>%s</th>\n" % cat)
        f.write("<th></th>\n")  # titles
        f.write("</tr>\n")

        f.write("<tr>\n")
        # 3D
        f.write("\t<td rowspan=\"2\">\n")
        path_im = os.path.relpath(www_grid['3d'], path_www)
        f.write("\t<a href=\"%s\">\n"
                "\t\t<img src=\"%s\" height=\"400\" />\n"
                "\t</a>\n" % (path_im, path_im))

        # Evidence sum
        f.write("\t<td rowspan=\"2\">\n")
        path_im = os.path.relpath(www_grid['o_sum'], path_www)
        f.write("\t<a href=\"%s\">\n"
                "\t\t<img src=\"%s\" height=\"400\" />\n"
                "\t</a>\n" % (path_im, path_im))
        # Evidence
        for cat in www_grid['evidence']:
            f.write("<td style=\"padding-bottom: 2px\">\n")
            path_im = os.path.relpath(www_grid['evidence'][cat], path_www)
            f.write("\t<a href=\"%s\">\n"
                    "\t\t<img src=\"%s\" height=\"200\" />\n"
                    "\t</a>\n" % (path_im, path_im))
            f.write("</td>\n")
        f.write("<td>Evidence</td>\n")

        f.write("\t</td>\n")
        f.write("</tr>\n")

        f.write("<tr>\n")
        for cat in www_grid['occ']:
            f.write("<td>\n")
            path_im = os.path.relpath(www_grid['occ'][cat], path_www)
            f.write("\t<a href=\"%s\">\n"
                    "\t\t<img src=\"%s\" height=\"200\" />\n"
                    "</a>\n" % (path_im, path_im))
            f.write("</td>\n")
        f.write("<td>Occupancy map</td>\n")
        f.write("</tr>")

        f.write('</table>')

    return True


if __name__ == '__main__':
   main(sys.argv[1:])
