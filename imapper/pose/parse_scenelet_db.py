"""

"""
from imapper.visualization.plotting import plt
import argparse
import sys
import os
import numpy as np
from math import ceil, floor
from imapper.util.stealth_logging import lg
from imapper.util.my_argparse import argparse_check_exists
from imapper.logic.scenelet import Scenelet
from imapper.pose.match_gap import read_scenelets
from imapper.util.my_pickle import pickle_load, pickle
from imapper.scenelet_fit.main_candidates_paul import read_charness


def is_close(scene_obj, skeleton, dist_thresh, return_dist=False):
    if scene_obj.label == 'floor':
        return True
    N_JOINTS = skeleton.N_JOINTS
    # _point_close_to_obb = scene_obj.point_close_to_obb
    _get_joint_3d = skeleton.get_joint_3d
    frame_ids = skeleton.get_frames()
    # for pose in scenelet.skeleton.poses.values():
    dist = min(part.obb.closest_face_dist_memoized(
      _get_joint_3d(_joint_id, _frame_id))
               for _frame_id, _joint_id in ((_frame_id, _joint_id)
                                            for _frame_id in frame_ids
                                            for _joint_id in range(N_JOINTS))
               for part in scene_obj.parts.values())
    if return_dist:
        return dist < dist_thresh, dist
    else:
        return dist < dist_thresh


def plot_charness(scenes, d_dir, ch_thresh=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    charnesses = [s.charness for s in scenes if s.charness > ch_thresh]
    ax.hist(charnesses, bins=30, cumulative=True)
    plt.title("Cumulative charness above %g" % ch_thresh)
    p = os.path.join(d_dir, 'charnesses.png')
    plt.savefig(p, dpi=200)
    lg.debug("Saved charnesses to %s" % p)
    plt.close()

_blacklist = [("scene10_take1-11620-12570__scenelet_15", []),
              ("scene09_take1-00725-01175__scenelet_8", ["01_table_top"]),
              ("scene10_take1-10590-11530__scenelet_13", ["01_table_top"]),
              ("scene10_take1-00050-00820__scenelet_2", ["02_table_top"]),
              ("scene10_take1-00050-00820__scenelet_11", ["01_table_top"]),
              ("scene10_take1-06310-07030__scenelet_5", []),
              ("scene10_take1-11620-12570__scenelet_13", ["01_table_top"]),
              ("gates503_mati1_2014-05-19-18-56-49__scenelet_17", [])]

def main(argv):
    pjoin = os.path.join  # cache long name
    parser = argparse.ArgumentParser(
      "Find characteristic scene times",
      description="Scans a directory of short scenelets (exported from "
                  "Matlab), and looks up their full version in the original "
                  "scenes. It exports a new scenelet containing all poses "
                  "between the start and end times of the input short "
                  "scenelets. Matching is done by name."
                  "Scenelets below time length limit and not enough objects "
                  "are thrown away."
                  "It also saves the scenelet characteristicness into the "
                  "output scenelet files.")
    parser.add_argument(
      'd', type=argparse_check_exists,
      help="Folder containing PiGraphs scenelets. E.g. "
           "/mnt/thorin_data/stealth/shared/"
           "pigraph_scenelets__linterval_squarehist_large_radiusx2")
    parser.add_argument(
      's', type=argparse_check_exists,
      help="Folder containing PiGraphs full scenes. E.g. "
           "/mnt/thorin_data/stealth/shared/scenes_pigraphs")
    parser.add_argument(
      '-l', '--limit-len', type=int,
      help="Minimum length for a scenelet",
      default=10)  # changed from `5` on 15/1/2018
    parser.add_argument(
      '--dist-thresh', type=float,
      help='Distance threshold for object pruning. Typically: 0.2 or 0.5.',
      default=.5
    )
    # parse arguments
    args = parser.parse_args(argv)
    parts_to_remove = ['sidetable']
    lg.warning("Will remove all parts named %s" % parts_to_remove)

    # read scenes and scenelets
    p_pickle = pjoin(args.d, 'scenes_and_scenelets.pickle')
    if os.path.exists(p_pickle):
        lg.info("reading from %s" % p_pickle)
        scenes, scenelets = pickle_load(open(p_pickle, 'rb'))
        lg.info("read from %s" % p_pickle)
    else:
        scenelets = read_scenelets(args.d)
        scenes = read_scenelets(args.s)
        scenes = {scene.name_scene: scene for scene in scenes}
        pickle.dump((scenes, scenelets), open(p_pickle, 'wb'), protocol=-1)
        lg.info("wrote to %s" % p_pickle)

    # Read characteristicnesses (to put them into the scenelet).
    p_charness = pjoin(args.d, "charness__gaussian.mat")
    pose_charness, scenelet_names = read_charness(p_charness,
                                                  return_hists=False,
                                                  return_names=True)

    # output folder
    d_scenelets_parent = os.path.dirname(args.d)
    d_dest = pjoin(d_scenelets_parent, 'deb',
                   "%s_full_sampling" % args.d.split(os.sep)[-1])
    # makedirs_backed
    if os.path.exists(d_dest):
        i = 0
        while i < 100:
            try:
                os.rename(d_dest, "%s.bak.%02d" % (d_dest, i))
                break
            except OSError:
                i += 1
    os.makedirs(d_dest)

    # _is_close = is_close  # cache namespace lookup
    # processing
    for sclt in scenelets:
        # cache skeleton
        skeleton = sclt.skeleton

        if 'scene09' in sclt.name_scenelet or 'scene10' in sclt.name_scenelet:
            lg.debug("here")
        else:
            continue

        # prune objects
        per_cat = {}
        cnt = 0
        for oid, scene_obj in sclt.objects.items():
            close_, dist = is_close(scene_obj, skeleton, args.dist_thresh,
                                    return_dist=True)
            label = scene_obj.label
            if 'chair' in label or 'couch' in label or 'stool' in label:
                label = 'sittable'

            try:
                per_cat[label].append((dist, oid))
            except KeyError:
                per_cat[label] = [(dist, oid)]
            if scene_obj.label != 'floor':
                cnt += 1

        per_cat = {k: sorted(v) for k, v in per_cat.items()}

        name_scene = sclt.name_scene.split('__')[0]
        if '-no-coffeetable' in name_scene:
            name_scene = name_scene[:name_scene.find('-no-coffeetable')]
        scene = scenes[name_scene]

        if 'shelf' not in per_cat:
            for oid, ob in scene.objects.items():
                if ob.label == 'shelf':
                    close_, dist = is_close(ob, skeleton, args.dist_thresh,
                                            return_dist=True)
                    oid_ = oid
                    while oid_ in sclt.objects:
                        oid_ += 1
                    sclt.add_object(oid_, ob)
                    cnt += 1
                    try:
                        per_cat['shelf'].append((dist, oid_))
                    except KeyError:
                        per_cat['shelf'] = [(dist, oid_)]

        if 'shelf' in per_cat:
            assert len(per_cat['shelf']) == 1, "TODO: keep all shelves"

        oids_to_keep = [v[0][1] for v in per_cat.values()
                        if v[0][0] < args.dist_thresh]

        if not len(oids_to_keep):  # there is always a floor
            lg.warning("Skipping %s, not enough objects: %s"
                       % (sclt.name_scenelet, per_cat))
            continue

        # if 'gates392_mati3_2014-04-30-21-13-46__scenelet_25' \
        #     == sclt.name_scenelet:
        #     lg.debug("here")
        # else:
        #     continue

        # copy skeleton with dense sampling in time
        mn, mx = skeleton.get_frames_min_max()
        # assert mn == 0, "This usually starts indexing from 0, " \
        #                 "no explicit problem, just flagging the change."
        time_mn = floor(skeleton.get_time(mn))
        time_mx = ceil(skeleton.get_time(mx))

        # artificially prolong mocap scenes
        if 'scene' in name_scene and (time_mx - time_mn < 60):
            d = (time_mx - time_mn) // 2 + 1
            time_mn -= d
            time_mx += d
        # lookup original scene name
        # mn_frame_id_scene, mx_frame_id_scene = \
        #     scene.skeleton.get_frames_min_max()

        frame_ids_old = skeleton.get_frames()
        times_old = [skeleton.get_time(fid) for fid in frame_ids_old]
        for frame_id in frame_ids_old:
            skeleton.remove_pose(frame_id)

        for frame_id in range(time_mn, time_mx+1):
            if not scene.skeleton.has_pose(frame_id):
                continue
            pose = scene.skeleton.get_pose(frame_id=frame_id)
            # scale mocap skeletons
            fw = scene.skeleton.get_forward(frame_id=frame_id,
                                            estimate_ok=False)
            sclt.set_pose(frame_id=frame_id, angles=None,
                          pose=pose, forward=fw, clone_forward=True)

        if 'scene0' in name_scene or 'scene10' in name_scene:
            mx_old = np.max(sclt.skeleton.poses[:, 1, :])
            sclt.skeleton.poses *= 0.8
            mx_new = np.max(sclt.skeleton.poses[:, 1, :])
            sclt.skeleton.poses[1, :] += mx_new - mx_old + 0.05
        _frames = sclt.skeleton.get_frames()

        # check length
        if len(_frames) < args.limit_len:
            lg.warning("Skipping %s, because not enough frames: %s"
                       % (sclt.name_scene, _frames))
            continue

        # save charness
        try:
            id_charness = next(i for i in range(len(scenelet_names))
                               if scenelet_names[i] == sclt.name_scenelet)
            sclt.charness = pose_charness[id_charness]
        except StopIteration:
            lg.error("Something is wrong, can't find %s, %s in charness db "
                     "containing names such as %s."
                     % (sclt.name_scene, sclt.name_scenelet,
                        scenelet_names[0]))
            sclt.charness = 0.4111111

        _mn, _mx = (_frames[0], _frames[-1])
        assert _mn >= time_mn, "not inside? %s < %s" % (_mn, time_mn)
        assert _mx <= time_mx, "not inside? %s < %s" % (_mx, time_mx)
        if len(_frames) < len(frame_ids_old):
            lg.warning("Not more frames than interpolated "
                       "scenelet?\n%s\n%s\n%s"
                       % (_frames, frame_ids_old, times_old))

        oids = list(sclt.objects.keys())
        for oid in oids:
            if oid not in oids_to_keep:
                lg.debug("removed %s" % sclt.objects[oid])
                sclt.objects.pop(oid)
            else:
                obj = sclt.objects[oid]
                part_ids_to_remove = [part_id
                                      for part_id, part in obj.parts.items()
                                      if part.label in parts_to_remove]
                if len(part_ids_to_remove) == len(obj.parts):
                    sclt.objects.pop(oid)
                else:
                    for part_id in part_ids_to_remove:
                        lg.debug("removed %s" % sclt.objects[obj].parts[part_id])
                        obj.parts.pop(part_id)
        if len(sclt.objects) < 2 and next(iter(sclt.objects.values())).label == 'floor':
            lg.debug("finally removing scenelet: %s" % sclt.objects)
            continue

        # save in the scene folder
        d_dest_scene = pjoin(d_dest, name_scene)
        if not os.path.exists(d_dest_scene):
            os.makedirs(d_dest_scene)
        sclt.save(pjoin(d_dest_scene, "skel_%s" % sclt.name_scenelet))


if __name__ == '__main__':
    main(sys.argv[1:])


