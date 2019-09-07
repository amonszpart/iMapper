import numpy as np

from imapper.util.stealth_logging import lg
from imapper.logic.scenelet import Scenelet
from imapper.logic.scene_object import SceneObj, Obb
from imapper.logic.categories import CATEGORIES
from imapper.logic.joints import Joint


def export_scenelet(um, o_pos_3d, o_polys_3d, query_full_skeleton,
                    scenes, joints_active, transform_id=None):
    """Extract a scenelet (poses and objects) from the data from the
    optimized problem.

    Args:
        um (stealth.pose.unk_manager.UnkManager):
            Data manager.
        o_pos_3d (np.ndarray):
            Output 3D poses.
        o_polys_3d (np.ndarray): (6K, 4, 3)
            3D oriented bounding boxes stored stacked.
        query_full_skeleton (stealth.logic.skeleton.Skeleton):
            Initial path containing time information.
        joints_active (list):
            List of joint_ids that were optimized for.
            Usage: pose16[:, joints_active] = o_pos_3d[pid, :, :]
        transform_id (int):
            Export only a specific group. Everything is exported, if None.
    Returns:
        A scenelet extracted from the data provided.
    """
    # cache function
    _guess_time_at = query_full_skeleton.guess_time_at

    # all poses or the ones that belong to a group/scenelet
    if transform_id is None:
        pids_sorted = sorted([(pid, pid2scene)
                              for pid, pid2scene in um.pids_2_scenes.items()],
                             key=lambda e: e[1].frame_id)
    else:
        # pids_sorted = sorted([(pid, pid2scene)
        #                       for pid, pid2scene in um.pids_2_scenes.items()
        #                       if pid2scene.transform_id == transform_id],
        #                      key=lambda e: e[1].frame_id)
        pids_2_scenes = um.pids_2_scenes
        pids_sorted = sorted([(pid, pids_2_scenes[pid])
                              for pid in um.get_pids_for(transform_id)],
                             key=lambda e: e[1].frame_id)

    # create output scenelet
    o = Scenelet()
    charness = None

    #
    # Skeleton
    #

    # cache skeleton reference
    skeleton = o.skeleton
    # fill skeleton
    for pid, pid2scene in pids_sorted:
        if charness is None:
            scene = scenes[pid2scene.id_scene]
            charness = scene.charness
            o.add_aux_info('name_scenelet', scene.name_scenelet)
            o.charness = charness
        # get frame_id
        frame_id = int(pid2scene.frame_id)
        # check if already exists
        if skeleton.has_pose(frame_id):
            # TODO: fix overlapping frame_ids
            lg.warning("[export_scenelet] Overwriting output frame_id %d"
                       % frame_id)
        # add with time guessed from input skeleton rate
        pose = np.zeros((3, Joint.get_num_joints()))
        pose[:, joints_active] = o_pos_3d[pid, :, :]
        pose[:, Joint.PELV] = (pose[:, Joint.LHIP]
                               + pose[:, Joint.RHIP]) / 2.
        pose[:, Joint.NECK] = (pose[:, Joint.HEAD]
                               + pose[:, Joint.THRX]) / 2.
        # for j, jid in joints_remap.items():
        #     pose[:, j] = o_pos_3d[pid, :, jid]
        assert not skeleton.has_pose(frame_id=frame_id), \
            'Already has pose: {}'.format(frame_id)
        skeleton.set_pose(frame_id=frame_id,
                          pose=pose,
                          time=_guess_time_at(frame_id))

    #
    # Objects
    #

    scene_obj = None
    scene_obj_oid = 0  # unique identifier that groups parts to objects
    for polys2scene in um.polys2scene.values():
        # Check, if we are restricted to a certain group
        if transform_id is not None \
          and polys2scene.transform_id != transform_id:
            continue
        start = polys2scene.poly_id_start
        end = start + polys2scene.n_polys

        # 6 x 4 x 3
        polys = o_polys_3d[start:end, ...]
        assert polys.shape[0] == 6, "Assumed cuboids here"
        if scene_obj is None or scene_obj_oid != polys2scene.object_id:
            category = next(cat for cat in CATEGORIES
                            if CATEGORIES[cat] == polys2scene.cat_id)
            scene_obj = SceneObj(label=category)
            scene_obj_oid = polys2scene.object_id
            o.add_object(obj_id=-1, scene_obj=scene_obj, clone=False)
        part = scene_obj.add_part(part_id=-1,
                                  label_or_part=polys2scene.part_label)
        # TODO: average for numerical precision errors
        centroid = np.mean(polys, axis=(0, 1))
        ax0 = polys[0, 1, :] - polys[0, 0, :]
        scale0 = np.linalg.norm(ax0)
        ax0 /= scale0
        ax1 = polys[0, 3, :] - polys[0, 0, :]
        scale1 = np.linalg.norm(ax1)
        ax1 /= scale1
        ax2 = polys[1, 0, :] - polys[0, 0, :]
        scale2 = np.linalg.norm(ax2)
        ax2 /= scale2
        part.obb = Obb(centroid=centroid,
                       axes=np.concatenate((
                           ax0[:, None], ax1[:, None], ax2[:, None]
                       ), axis=1),
                       scales=[scale0, scale1, scale2])
    # if scene_obj is not None:
    #     o.add_object(obj_id=-1, scene_obj=scene_obj, clone=False)
    # else:
    #     lg.warning("No objects in scenelet?")

    # scene_obj = SceneObj('couch')
    # for poly_id in range(0, o_polys_3d.shape[0], 6):
    #     rects = o_polys_3d[poly_id : poly_id + 6, ...]
    #     # lg.debug("rects:\n%s" % rects)
    #     scene_obj.add_part(poly_id, 'seat')
    #
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # for rid, rect in enumerate(rects):
    #     #     wrapped = np.concatenate((rect, rect[0:1, :]), axis=0)
    #     #     ax.plot(wrapped[:, 0], wrapped[:, 2], wrapped[:, 1])
    #     #     for ci in range(4):
    #     #         c = rect[ci, :]
    #     #         ax.text(c[0], c[2], c[1], s="%d, %d, %d"
    #     #                                     % (poly_id, rid, ci))
    #     #     if rid >= 1:
    #     #         break
    #     #
    #     # plt.show()
    #     part = scene_obj.get_part(poly_id)
    #     centroid = np.mean(rects, axis=(0, 1))
    #     ax0 = rects[0, 1, :] - rects[0, 0, :]
    #     scale0 = np.linalg.norm(ax0)
    #     ax0 /= scale0
    #     ax1 = rects[0, 3, :] - rects[0, 0, :]
    #     scale1 = np.linalg.norm(ax1)
    #     ax1 /= scale1
    #     ax2 = rects[1, 0, :] - rects[0, 0, :]
    #     scale2 = np.linalg.norm(ax2)
    #     ax2 /= scale2
    #     part.obb = Obb(centroid=centroid,
    #                    axes=np.concatenate((
    #                        ax0[:, None], ax1[:, None], ax2[:, None]
    #                    ), axis=1),
    #                    scales=[scale0, scale1, scale2])
    # o.add_object(obj_id=99, scene_obj=scene_obj,
    #                     clone=False)

    return o
