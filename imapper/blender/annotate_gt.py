import bpy
import os
import sys
import numpy as np
from typing import Callable

path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir,
                                    os.pardir, os.pardir))
if path not in sys.path:
    sys.path.insert(0, path)
import imapper.logic.skeleton
import importlib
importlib.reload(imapper.logic.skeleton)

from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.logic.scene_object import SceneObj, SceneObjPart, Obb
from imapper.logic.joints import Joint


def from_blender(v):
    return [v[0], -v[2], v[1]]


def extract_annotated_scenelet(scene, prefix_obj='obb',
                               frame_ids=None,
                               frame_multiplier=1., time_multiplier=1.,
                               f_ob_is_joint=lambda ob:
                                 ob.name.startswith('Output')
                                 and ob.name.endswith('Sphere'),
                               f_joint_name_from_ob=lambda ob:
                                 ob.name.split('.')[1]
                               ):
    """
    
    Args:
        scene (bpy.types.Scene):
            The current scene (e.g. bpy.context.scene).
        prefix_obj (str):
            Start of object names that we want to include in the scenelet
            as oriented bounding boxes.
        frame_ids (List[int]):
            A subset of frame IDs to export.
        frame_multiplier (float):
            Scaling for frame IDs. The result will be rounded and truncated.
            output.frame_id := int(round(frame_id * frame_multiplier))
        time_multipler (float):
            Scaling for times associated with frame_ids.
            output.time := int(round(frame_id * frame_multiplier)) 
            * time_multiplier.
        f_ob_is_joint (Callable[[bpy.types.Object], bool]]):
            Decides if a Blender object is a joint.
        f_joint_name_from_ob (Callable[[bpy.types.Object], str]):
            Gets the joint name from the Blender object name.
    """
    # joints = {
    #     ob.name.split('.')[1]: ob
    #     for ob in bpy.data.objects
    #     if ob.name.startswith('Output') and ob.name.endswith('Sphere')}
    joints = {
        f_joint_name_from_ob(ob): ob
        for ob in bpy.data.objects
        if f_ob_is_joint(ob)}
    print("joints: %s" % joints)
    skeleton = Skeleton()
    if len(joints):
        assert len(joints) == 16, "No: %s" % len(joints)
        if not frame_ids:
            frame_ids = range(scene.frame_start, scene.frame_end + 1)
        for frame_id in frame_ids:
            o_frame_id = int(round(frame_id * frame_multiplier))
            if skeleton.has_pose(o_frame_id):
                print("skipping %s" % frame_id)
                continue
            print("frame_id: %s" % frame_id)
            scene.frame_set(frame_id)
            bpy.context.scene.update()
            # bpy.ops.anim.change_frame(frame_id)
            pose = np.zeros(shape=(3, len(joints)))
            for joint, ob in joints.items():
                pos = ob.matrix_world.col[3]
                print("pos[%s]: %s" % (ob.name, pos))
                joint_id = Joint.from_string(joint)
                print("joint %s is %s" % (joint, Joint(joint_id)))
                pose[:, joint_id] = from_blender(pos)
            print("o_frame: %s from %s" % (o_frame_id, frame_id))
            assert not skeleton.has_pose(o_frame_id), \
                "Already has %s" % frame_id
            skeleton.set_pose(frame_id=o_frame_id, pose=pose,
                              time=o_frame_id * time_multiplier)
    objs_bl = {}
    for obj in bpy.data.objects:
        if obj.name.startswith(prefix_obj) and not obj.hide:
            obj_id = int(obj.name.split('_')[1])
            try:
                objs_bl[obj_id].append(obj)
            except KeyError:
                objs_bl[obj_id] = [obj]

    print("objs: %s" % objs_bl)
    scenelet = Scenelet(skeleton=skeleton)
    print("scenelet: %s" % scenelet)
    for obj_id, parts_bl in objs_bl.items():
        name_category = None
        scene_obj = None
        for part_id, part_bl in enumerate(parts_bl):
            transl, rot, scale = part_bl.matrix_world.decompose()
            rot = rot.to_matrix()
            if any(comp < 0. for comp in scale):
                scale *= -1.
                rot *= -1.
            assert not any(comp < 0. for comp in scale), "No: %s" % scale

            matrix_world = part_bl.matrix_world.copy()

            # need to save full scale, not only half axes
            for c in range(3):
                for r in range(3):
                    matrix_world[r][c] *= 2.
            name_parts = part_bl.name.split('_')
            if name_category is None:
                name_category = name_parts[2]
                scene_obj = SceneObj(label=name_category)
            else:
                assert name_category == name_parts[2], \
                    "No: %s %s" % (name_category, name_parts[2])
            name_part = name_parts[3]
            print("part: %s" % name_part)
            part = SceneObjPart(name_part)
            part.obb = Obb(
              centroid=np.array(from_blender([transl[0], transl[1], transl[2]])),
              axes=np.array([
                  [rot[0][0], rot[0][1], rot[0][2]],
                  [-rot[2][0], -rot[2][1], -rot[2][2]],
                  [rot[1][0], rot[1][1], rot[1][2]]
              ]),
              scales=np.array([scale[0] * 2., scale[1] * 2., scale[2] * 2.])
            )
            # if 'table' in name_category:
            #     print(part.obb.axes)
            #     raise RuntimeError("stop")
            print("obb: %s" % part.obb.to_json(0))
            scene_obj.add_part(part_id, part)
        scenelet.add_object(obj_id, scene_obj, clone=False)
    return scenelet


def save_annotated_scenelet(sc_path, scene, prefix_obj='obb',
                            frame_ids=None,
                            frame_multiplier=1., time_multiplier=1.):
    """Ground truth annotation function."""
    path_out = os.path.normpath(bpy.path.abspath(sc_path))
    print("path_out: %s" % path_out)
    print("scene: %s" % bpy.context.scene)
    for s in bpy.data.scenes:
        print("scene: %s" % s)
        print("e: %s" % s.render.engine)
    for ob in bpy.data.objects:
        print(ob)
    scenelet = extract_annotated_scenelet(
      scene=scene, prefix_obj=prefix_obj, frame_ids=frame_ids,
      frame_multiplier=frame_multiplier, time_multiplier=time_multiplier)
    scenelet.save(path=path_out, save_obj=True)
    return scenelet

def add_constraint_to_spheres():
    print("here")
    bpy.ops.object.select_all(action='DESELECT')
    rig = next(ob for ob in bpy.data.objects if ob.name == 'Output.Rig')
    amt = bpy.data.armatures['Output.RigAmt']
    for ob in bpy.data.objects:
        if ob.parent and ob.parent.name == 'Output.Actor.Group' and ob.name.endswith('Sphere'):
            print("ob: %s" % ob.name)
            joint = ob.name.split('.')[1]
            if not len(ob.constraints):
                #ob.select = True
                bpy.ops.object.select_pattern(pattern=ob.name, extend=False)
                for ob2 in bpy.data.objects:
                    if ob2.select:
                        print("ob selected: %s" % ob2.name)
                print("joint: %s" % joint)
                #bpy.ops.object.constraint_add(type='COPY_LOCATION')
                cnstr = ob.constraints.new(type='COPY_LOCATION')
            else:
                cnstr = ob.constraints[0]
                #cnstr = bpy.context.object.constraints['Copy Location']
                print("cnstr: %s" % cnstr)
                cnstr.target = rig
                cnstr.subtarget = "Output.Rig.%s" % joint
                #if 'ANK' in joint or 'KNE' in joint or 'SHO' in joint or 'ELB' in joint or 'WRI' in joint:
                cnstr.head_tail = 1.
            ob.hide_select = True
        elif ob.name.endswith('Cylinder'):
            ob.hide = True

def find_keyframes(filter_objects=None):
    """Collects all integer keyframes in the Blender scene.

    Args:
        filter_objects (function):
            Takes a blender object as parameter and returns true, if we are
            interested in its keyframes.
    Returns:
        A set of integer frame_ids that have keyframes from at least one of
        the objects in the scene that the filter_objects function returned
        true for, if filter function was provided.
    """
    frame_ids = set()
    for ob in bpy.data.objects:
        if filter_objects and not filter_objects(ob):
            continue
        if not ob.animation_data:
            continue
        fcurves = ob.animation_data.action.fcurves
        for fc in fcurves:
            for kf in fc.keyframe_points:
                frame_ids.add(kf.co[0])
    return frame_ids

def prune_keyframes(args):
    keyframes = [[kf, None, None] for kf in args]
    print(keyframes)
    keep = None
    for ob in bpy.data.objects:
        if ob.name.startswith('Output.'):
            if not ob.animation_data:
                continue
            fcurves = ob.animation_data.action.fcurves
            if keep is None:
                for fc in fcurves:
                    for kf in fc.keyframe_points:
                        print(kf.co)
                        for i in range(len(keyframes)):
                            print("kf[i]: %s" % repr(keyframes[i]))
                            diff = abs(keyframes[i][0] - kf.co[0])
                            if keyframes[i][1] is None or diff < keyframes[i][1]:
                                keyframes[i][1] = diff
                                keyframes[i][2] = int(kf.co[0])
                keep = [f[2] for f in keyframes]

            for fc in fcurves:
                to_rem = []
                for kf in fc.keyframe_points:
                    if int(kf.co[0]) not in keep:
                        print("would remove %d, because not in %s"
                              % (int(kf.co[0]), keep))
                        # fc.keyframe_points.remove(kf)
                        to_rem.append(kf)
                for kf in to_rem:
                    print("kf: %s" % kf)
                    try:
                        fc.keyframe_points.remove(kf)
                        bpy.context.scene.update()
                    except RuntimeError:
                        pass

def compare_scenes(sclt_ours, sclt_gt, object_correspondences,
                   up=np.array((0., -1., 0.))):
    np.set_printoptions(suppress=True)
    accum = 0.
    cnt = 0
    diffs = []
    up = None
    for corr in object_correspondences:
        print("working with %s" % repr(corr))
        # print("11: %s" % sclt_ours.objects[11].get_part_by_name_strict(
        #   'seat').obb.centroid)
        ours = corr[0] if isinstance(corr[0], list) \
                else [corr[0]]
        gt = corr[1] if isinstance(corr[1], list) \
                else [corr[1]]
        
        # manual centroid from inspection
        if len(ours) == 1 and len(ours[0]) == 3:
            c0 = np.array(ours[0])[:, None]
        else:
            c0 = [
                sclt_ours.objects[e[0]].get_part_by_name_strict(e[1]).obb.centroid
                for e in ours]
            c0 = np.mean(c0, axis=0)
        # print("c0: %s" % c0)
        # for e in gt:
        #     ob = sclt_gt.objects[e[0]]
            # print(e[0], ob)
            # for oid, p in ob.parts.items():
            #     print(oid, p.label)
            # print(ob.get_part_by_name_strict(e[1]).obb.centroid)
        c1 = [sclt_gt.objects[e[0]].get_part_by_name_strict(e[1]).obb.centroid
              for e in gt]
        c1 = np.mean(c1, axis=0)
        print('ours: {}, gt: {}'.format(c0, c1))
        # print("c1: %s" % c1)

        if up is None:
            up = sclt_gt.objects[gt[0][0]] \
                .get_part_by_name_strict(gt[0][1]).obb._axes[1]
            up /= np.linalg.norm(up)
            print("up(%s): %s" % (up, sclt_gt.objects[gt[0][0]].label))
        acos_deg = np.rad2deg(np.arccos(abs(up.dot((0., 0., -1.)))))
        assert abs(acos_deg) < 10., \
            "Up vector not really up? %s, %s deg" % (up, acos_deg)
        world_up = np.array(from_blender(up))[None, :]

        # NOTE: this is not really true, only works for z-up
        # print("world_up: %s" % world_up)
        diff = c1 - c0
        # print("diff: %s" % diff)
        dot = np.dot(diff.T, world_up.T)
        # print(dot)
        diff -= dot
        norm = np.linalg.norm(diff)
        print("diff(%s, %s): %s (%s)\n\t%s - %s"
              % (corr[0], corr[1], diff.T, norm, c0.T, c1.T))
        accum += norm
        cnt += 1
        diffs.append(norm)
    print("Avgdiff: %s" % (accum / cnt))
    print("diffs: %s" % " ".join(["%g" % d for d in diffs]))

def get_correspondences(name_scene, postfix=""):
    l = None
    if name_scene == 'lobby15':
        l = [
            (
                [(5, 'seat'), (8, 'seat')], # ours
                (0, 'seat') # gt
            ), (
                [(7, 'seat'), (9, 'seat')], # ours
                (1, 'seat') # gt
            ), (
                (0, 'top'), # ours
                (2, 'top') #gt
            )
        ]
    elif name_scene == 'lobby19-3':
        if postfix == 'gt':
            l = [
                (
                    [(11, 'seat'), (2, 'seat'), (6, 'seat')],
                     (0, 'seat')),
                 ((7, 'legs'), (1, 'top'))]
        elif postfix == 'rcnn':
            l = [
                (
                    [(0, 'seat')], # other
                    (0, 'seat')    # gt
                ),
                  (
                  [(1, 'top')], # other
                  (1, 'top')    # gt
                )
            ]
    elif name_scene == 'lobby22-1':
        l = [
            (    # closest couch
                [(0, 'seat'), (3, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # right couch
                [(10, 'seat'), (12, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # farthest couch
                [(5, 'legs'), (13, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # table
                (8, 'top'),  # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'gates250_mati1_2014-05-23-21-17-59':
        l = [
            (    # closest couch
                [(0, 'seat'), (4, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # right couch
                [(3, 'seat'), (5, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # table
                (7, 'top'),  # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'garden1':
        l = [
            (    # left couch
                [(3, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # chair 1 on right
                [(7, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # chair 2 on right
                [(5, 'seat')],  # ours
                (2, 'seat')  # gt
            ), ( # back chair
                (13, 'seat'),  # ours
                (3, 'seat')  # gt
            ), ( # table
                (0, 'top'),  # ours
                (4, 'top')  # gt
            )
        ]
    elif name_scene == 'library3':
        l = [
            (    # left shelf
                [(16, 'shelf')],  # ours
                (0, 'shelf')  # gt
            ), ( # right shelf
                [(3, 'shelf')],  # ours
                (2, 'shelf')  # gt
            ), ( # right blue chair aka. single couch
                [(13, 'seat')],  # ours
                (9, 'seat')  # gt
            ), ( # right chair out of closest chairs
                [(5, 'seat')],  # ours
                (5, 'seat')  # gt
            ), ( # left chair out of closest chairs
                [(1, 'seat')],  # ours
                (8, 'seat')  # gt
            ), ( # left couch
                [(2, 'seat'), (6, 'seat')],  # ours
                (4, 'seat')  # gt
            )
        ]
    elif name_scene == 'lobby18-1':
        l = [
            (
                [(1, 'seat'), (3, 'seat')],  # ours
                (0, 'seat')  # gt
            )
        ]
    elif name_scene == 'lobby11-couch':
        l = [
            (
                [(0, 'seat'), (1, 'seat')],  # ours
                (0, 'seat')  # gt
            )
        ]
    elif name_scene == 'lobby12-couch-table':
        l = [
            (    #
                [(1, 'seat'), (1, 'seat-1')],  # ours
                (0, 'seat')  # gt
            ),
            (
                (0, 'top'), # ours
                (1, 'top')  # gt
            )
        ]
    elif name_scene == 'lobby24-3-2':
        l = [
            (    # back couch
                [(0, 'seat'), (3, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # right couch
                [(2, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # front couch
                [(1, 'seat'), (4, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # table
                (15, 'top'), # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'lobby24-3-1': # 2 sits
        l = [
            (    # back couch
                [(0, 'seat'), (4, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # right couch
                [(5, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # table
                (7, 'top'), # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'lobby24-3-3': # 2 sits
        l = [
            (    # back couch
                [(0, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # table
                (1, 'top'), # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'lobby24-2-2':
        l = [
            (    # back couch
                [(7, 'seat'), (9, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # right couch
                [(2, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # front couch
                [(3, 'seat'), (6, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # table
                (0, 'top'), # ours
                (2, 'top')  # gt
            )
        ]
    elif name_scene == 'livingroom00':
        l = [
            (   # left couch
                [(2, 'seat')],  # ours
                (0, 'seat')  # gt
            ), ( # opposite couch, left seat
                [(5, 'seat')],  # ours
                (1, 'seat')  # gt
            ), ( # opposite couch, right seat
                [(4, 'seat')],  # ours
                (1, 'seat-1')  # gt
            ), ( # right couch
                [(1, 'seat')],  # ours
                (2, 'seat')  # gt
            ), ( # front chair
                [(7, 'seat')],  # ours
                (3, 'seat')  # gt
            ), ( # table
                (0, 'top'), # ours
                (4, 'top')  # gt
            )
        ]
    elif name_scene == 'office1-1':
        l = [
            (   # middle table
                [(4, 'top')],  # ours
                (16, 'top')  # gt
            ), (   # middle chair
                [(9, 'seat')],  # ours
                (3, 'seat')  # gt
            ), (   # middle chair
                [(14, 'shelf')],  # ours
                (8, 'shelf')  # gt
            ), (   # back table
                [(0, 'top')],  # ours
                (11, 'top')  # gt
            ), (   # back chair
                [(5, 'seat')],  # ours
                (7, 'seat')  # gt
            ), (   # front table
                [(17, 'top')],  # ours
                (15, 'top')  # gt
            ), (   # back chair
                [(16, 'seat')],  # ours
                (5, 'seat')  # gt
            )
        ]
    elif name_scene == 'office2-1':
        l = [
            (   # back left chair
                [(15, 'seat')],  # ours
                (26, 'seat')  # gt
            ), (   # back right chair
                [(5, 'seat')],  # ours
                (27, 'seat')  # gt
            ), (   # back left table
                [(11, 'top')],  # ours
                (12, 'top')  # gt
            ), (   # back right table
                [(19, 'top')],  # ours
                (2, 'top')  # gt
            ), (   # mid left chair
                [(18, 'stool')],  # ours
                (7, 'seat')  # gt
            ), (   # mid right chair
                [(0, 'seat')],  # ours
                (23, 'seat')  # gt
            ), (   # mid left table - read using 3D Cursor in Blender
                [(-0.433, 0.738, 5.677)],  # ours, (X, -Z, Y) from Blender
                (11, 'top')  # gt
            ), (   # mid right table - read using 3D Cursor in Blender
                [(0.96712, 0.73824, 4.81321)],  # ours, (X,-Z,Y) from Blender
                (10, 'top')  # gt
            ), (   # front left chair
                [(10, 'seat')],  # ours
                (24, 'seat')  # gt
            ), (   # front right chair
                [(4, 'seat')],  # ours
                (25, 'seat')  # gt
            ), (   # front left table
                [(6, 'top')],  # ours
                (22, 'top')  # gt
            ), (   # front right table
                [(9, 'top')],  # ours
                (19, 'top')  # gt
            )
        ]
    else:
        raise RuntimeError("[get_correspondences] Could not find scene by "
                           "name %s" % name_scene)

    return l


def get_frame_ids(name_scene):
    l = None
    if name_scene == 'lobby15':
        l = [119, 173, 187]
    elif name_scene == 'lobby19-3':
        l = [48, 54, 65]
    elif name_scene == 'lobby22-1':
        l = [40, 110, 169]
    elif name_scene == 'garden1':
        l = [109, 126, 135]
    elif name_scene == 'library3':
        l = [238, 246, 228]
    elif name_scene == 'lobby18-1':
        l = [39, 63, 105]
    elif name_scene == 'lobby11-couch':
        l = [1, 2, 3]
    elif name_scene == 'lobby12-couch-table':
        l = [1, 2, 3]
    elif name_scene == 'lobby24-3-1':
        l = [1, 2, 3]
    elif name_scene == 'lobby24-3-2':
        l = [1, 2, 3]
    elif name_scene == 'lobby24-3-3':
        l = [1, 2, 3]
    elif name_scene == 'lobby24-2-2':
        l = [1, 2, 3]
    elif name_scene == 'livingroom00':
        l = [1, 2, 3]
    elif name_scene == 'office1-1':
        l = [1, 2, 3]
    elif name_scene == 'office2-1':
        l = [1, 2, 3]
    else:
        raise RuntimeError("[get_frame_ids] scene not found: %s" % name_scene)

    return l

def compare_other_to_gt():
    scene = bpy.context.scene
    assert 'postfix' in globals(), "Need postfix parameter"
    name_scene = bpy.path.abspath('//').split(os.sep)[-3]
    if 'object_correspondences' not in globals():
        object_correspondences = get_correspondences(name_scene, postfix)
    p_gt = bpy.path.abspath("//../quant/skel_gt.json")
    sclt_gt = Scenelet.load(p_gt)

    p_other = bpy.path.abspath("//../quant/skel_%s.json" % postfix)
    if True or not os.path.exists(p_other):
        frame_ids = find_keyframes(lambda ob: ob.name.startswith('Output.'))
        sclt_other = save_annotated_scenelet(sc_path=p_other,
                                             scene=scene, frame_ids=frame_ids)
    else:
        sclt_other = Scenelet.load(p_other)
    compare_scenes(sclt_other, sclt_gt, object_correspondences)


if __name__ == 'add_constraint_to_spheres':
    add_constraint_to_spheres()
elif __name__ == 'save_annotated_scenelet':
    assert 'postfix' in globals(), "Need postfix parameter"
    print("abspath: %s" % bpy.path.abspath('//'))
    name_scene = bpy.path.abspath('//').split(os.sep)[-3]
    scene = bpy.context.scene
    
    # Aron commented out in Jan2019
    if 'object_correspondences' not in globals():
        object_correspondences = get_correspondences(name_scene)
    p_file = bpy.path.abspath('//../output/skel_output.json')
    assert os.path.exists(p_file), "Does not exist: %s" % p_file
    sclt_ours = Scenelet.load(p_file)
    p_out = bpy.path.abspath('//../quant/skel_output.json')
    sclt_ours.save(p_out, save_obj=True)

    p_gt = bpy.path.abspath("//../quant/skel_%s.json" % postfix)
    if not os.path.exists(p_gt):
        if 'frame_ids' not in globals():
            frame_ids = get_frame_ids(name_scene)
        # frame_ids = find_keyframes(lambda ob: ob.name.startswith('Output.'))
        sclt_gt = save_annotated_scenelet(sc_path=p_gt,
                                          scene=scene, frame_ids=frame_ids,
                                          frame_multiplier=frame_multiplier,
                                          time_multiplier=time_multiplier)
    else:
        print("NOT OVERWRITING GT")
        sclt_gt = Scenelet.load(p_gt)
        
    try:
        compare_scenes(sclt_ours, sclt_gt, object_correspondences)
    except TypeError as e:
        print("error: %s" % e)

    print("done")
elif __name__ == 'compare_other_to_gt':
    compare_other_to_gt()
elif __name__ == 'prune_keyframes':
    prune_keyframes(args)


# if True:
#    # Use your own script name here:
#    filename = "/home/amonszpa/workspace/stealth/scripts/stealth/blender/annotate_gt.py"
#
#    filepath = os.path.join(os.path.dirname(bpy.data.filepath), filename)
#    global_namespace = {"__file__": filepath, "__name__": "add_constraint_to_spheres"}
#    with open(filepath, 'rb') as file:
#        exec(compile(file.read(), filepath, 'exec'), global_namespace)
