from imapper.visualization.plotting import plt

import os

import numpy as np
from descartes import PolygonPatch
from shapely import geometry as geom

from imapper.logic.categories import CATEGORIES
from imapper.logic.colors import stealth_colors
from imapper.util.stealth_logging import lg


def show_output(tf_vars, deb_oo, deb_jo, d_query, session, smooth_pairs,
                d_postfix, f_postfix, um):
    colors = stealth_colors
    p_deb = os.path.join(d_query, 'debug_isec' + d_postfix)
    if not os.path.exists(p_deb):
        os.makedirs(p_deb)
    #     os.system("rm %s/*.png" % p_deb)
    #     os.system("rm %s/*.svg" % p_deb)
    # else:

    # assert np.allclose(tf_vars.oo_mask_same.eval(),
    #                    tf_vars.oo_mask_same_2.eval())
    # assert np.allclose(tf_vars.oo_mask_cat.eval(),
    #                    tf_vars.oo_mask_cat_2.eval())
    # assert np.allclose(tf_vars.oo_mask_interacting_2,
    #                    tf_vars._oo_mask_interacting.eval())
    # assert np.isclose(tf_vars._oo_mask_interacting_sum_inv.eval(),
    #                   tf_vars.oo_mask_interacting_sum_inv_2.eval())

    obj_vxs_t = tf_vars.obj_2d_vertices_transformed
    o_obj_vxs_t, o_joints, o_smooth_pairs, distances, o_mgrid_vxs_t = \
        session.run([obj_vxs_t, deb_jo['joints'], smooth_pairs,
                     deb_jo['d'], tf_vars._obj_2d_mgrid_vertices_transformed])
    o_polys, o_poly_indices = session.run(
      [tf_vars.obj_2d_polys, tf_vars._obj_2d_poly_transform_indices])
    o_oo_mask, o_d_oo = session.run([deb_oo['oo_mask'], deb_oo['d_oo']])
    py_cat_ids = np.squeeze(tf_vars.cat_ids_polys.eval(), axis=0)
    lg.debug("distances: %s" % repr(distances.shape))
    lg.debug("obj_vxs_t: %s" % repr(o_obj_vxs_t.shape))
    lg.debug("joints: %s" % repr(o_joints.shape))

    mn1 = np.min(o_obj_vxs_t, axis=0)[[0, 2]]
    mx1 = np.max(o_obj_vxs_t, axis=0)[[0, 2]]
    mn2 = np.min(o_smooth_pairs, axis=0)
    mn2 = np.minimum(mn2[:3], mn2[3:])[[0, 2]]
    mx2 = np.max(o_smooth_pairs, axis=0)
    mx2 = np.maximum(mx2[:3], mx2[3:])[[0, 2]]
    mn = np.minimum(mn1, mn2)
    mx = np.maximum(mx1, mx2)
    assert o_joints.shape[0] // 5 == distances.shape[0]
    o_joints = o_joints.reshape(5, -1, 3)[0, ...]
    assert o_polys.shape[0] == distances.shape[1]

    fig = plt.figure()
    ax0 = fig.add_subplot(111, aspect='equal')
    # for pair in o_smooth_pairs:
    #     ax0.plot([pair[0], pair[3]], [pair[2], pair[5]], 'k--')

    # for id_pnt in range(1, o_joints.shape[0]):
    #     pnt0 = o_joints[id_pnt, :]
    o_joints_ordered = np.asarray([
        e[1] for e in
        sorted([(um.pids_2_scenes[pid].frame_id, o_joints[pid, ...])
                for pid in range(o_joints.shape[0])],
               key=lambda e: e[0])])
    assert o_joints_ordered.ndim == 2 and o_joints_ordered.shape[1] == 3
    ax0.plot(o_joints_ordered[:, 0], o_joints_ordered[:, 2], 'kx--')

    for id_poly in range(distances.shape[1]):
        idx_t = o_poly_indices[id_poly, 0]
        color = tuple(c/255. for c in colors[(idx_t+1) % len(colors)])
        d_sum = 0.
        # poly = np.concatenate((
        #     o_obj_vxs_t[4*id_poly:4*(id_poly+1), :],
        #     o_obj_vxs_t[4*id_poly:4*id_poly+1, :]))

        # ax0.plot(o_polys[id_poly, :, 0], o_polys[id_poly, :, 2], color=color)
        shapely_poly = geom.asPolygon(o_polys[id_poly, :, [0, 2]].T)
        patch = PolygonPatch(shapely_poly,
                             facecolor=color, edgecolor=color, alpha=0.25)
        ax0.add_artist(patch)
        cat = next(cat for cat in CATEGORIES
                   if CATEGORIES[cat] == py_cat_ids[id_poly])
        xy = (shapely_poly.centroid.xy[0][0]-0.1,
              shapely_poly.centroid.xy[1][0])
        ax0.annotate(cat, xy=xy, color=color, fontsize=6)
        for id_pnt in range(distances.shape[0]):
            d_ = distances[id_pnt, id_poly]
            if d_ < 0.:
                pnt = o_joints[id_pnt, :]
                ax0.scatter(pnt[0], pnt[2], s=(5*(1+d_))**2, color=color,
                            zorder=5)
                d_sum += distances[id_pnt, id_poly]
                ax0.annotate("%.2f" % d_,
                             xy=(np.random.rand() * 0.1 + pnt[0] - 0.05,
                                 np.random.rand() * 0.1 + pnt[2] - 0.05),
                             fontsize=4)

        for id_pnt in range(o_d_oo.shape[0]):
            d_ = o_d_oo[id_pnt, id_poly]
            if d_ < 0.:
                pnt = o_mgrid_vxs_t[id_pnt, :]
                ax0.scatter(pnt[0], pnt[2], s=(7*(1+d_))**2, edgecolor=color,
                            zorder=5, color=(1., 1., 1., 0.))
                ax0.annotate("%.2f" % d_,
                             xy=(np.random.rand() * 0.1 + pnt[0] - 0.05,
                                 np.random.rand() * 0.1 + pnt[2] - 0.05),
                             fontsize=8, color='r', zorder=6)

    # fig.suptitle("d_sum: %s" % d_sum)
    p_out = os.path.join(p_deb, "op_%s.png" % f_postfix)
    plt.draw()
    ax0.set_xlim(mn[0]-0.2, mx[0]+0.2)
    ax0.set_ylim(mn[1]-0.2, mx[1]+0.2)
    # plt.show()

    try:
        fig.savefig(p_out, dpi=300)
        lg.debug("saved to %s" % p_out)
    except PermissionError as e:
        lg.error("Could not save %s" % e)

    plt.close(fig)