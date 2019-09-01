import argparse
import sys
import os
import bpy
import numpy as np

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.logic.joints import Joint
from imapper.blender.import_scene import import_skeleton_animation
from imapper.blender.annotate_gt import from_blender


def parse_args(argv):
    # remove blender arguments
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description='Interpolate missing frames.')
    parser.add_argument('path', type=str, default='./', help='Path to the json scene file.')
    parser.add_argument('--postfix', type=str, default='ikipol',
                        help='postfix for the output')
    # parser.add_argument('--output', type=str, default='', help='Path to the output blender file.')
    # parser.add_argument('--video', type=str, default='../input/input_cropped_10fps.mp4', help='Path to the input video, relative to the scene file path.')
    # parser.add_argument('--fps', type=int, default=10, help='Fps for the scene, this has to match the video fps if a video is given!')
    # parser.add_argument('--intrinsics', type=str, default='../intrinsics.json', help='Path to the json file containing the camera intrinsics, relative to the scene file path.')
    # parser.add_argument('--keypoints', type=str, default='', help='Path to the json file containing the 2d keypoints, relative to the scene file path (leave empty for default).')
    # parser.add_argument('--localpose', type=str, default='', help='Path to the json file containing the 2d keypoints, relative to the scene file path (leave empty for default).')
    # parser.add_argument('--render_width', type=int, default=1920, help='Width of the rendered image (height is determinded from video aspect ratio).')
    # # parser.add_argument('--render_height', type=int, default=1080, help='Height of the rendered image.')
    # parser.add_argument('--time_scale', type=float, default=1, help='Global scaling of animation time.')
    # parser.add_argument('--use_cycles', type=int, default=True, help='Use cycles renderer.')

    return parser.parse_args(argv)


def extract_skeleton(scene, frame_ids=None,
                     frame_multiplier=1., time_multiplier=1.):
    joints = {
        ob.name.split('.')[1]: ob
        for ob in bpy.data.objects
        if ob.name.startswith('Output') and ob.name.endswith('Sphere')}
    print("joints: %s" % joints)
    assert len(joints) == Joint.get_num_joints(), \
        "No: %s != %s" % (len(joints), Joint.get_num_joints())
    if not frame_ids:
        frame_ids = range(scene.frame_start, scene.frame_end+1)

    skeleton = Skeleton()
    for frame_id in frame_ids:
        o_frame_id = int(round(frame_id * frame_multiplier))
        if skeleton.has_pose(o_frame_id):
            print("skipping %s, because already set" % frame_id)
            continue
        print("frame_id: %s, o_frame_id: %s" % (frame_id, o_frame_id))
        scene.frame_set(frame_id)
        bpy.context.scene.update()
        pose = np.zeros(shape=(Skeleton.DIM, len(joints)))
        for joint, ob in joints.items():
            pos = ob.matrix_world.col[3]
            joint_id = Joint.from_string(joint)
            pose[:, joint_id] = from_blender(pos)
        assert not skeleton.has_pose(o_frame_id), "Already has %s" % frame_id
        skeleton.set_pose(frame_id=o_frame_id, pose=pose,
                          time=o_frame_id * time_multiplier)

    # scenelet = Scenelet(skeleton=skeleton)
    # scenelet.save(path=path_out, save_obj=False)

    return skeleton


def ipol_ik(p_scenelet, postfix):
    """Uses Blender's IK engine to interpolate 3D joint positions in time.

    Sphere positions are tied to Rig endpoints, and these endpoints
    will be as close as possible to the noisy targets. Hence the Sphere
    positions are the ones we save as interpolated 3D joint positions.

    The method will try to preserve all, non-skeleton related
    information in the scenelet by only overwriting its skeleton.
    Visibility and confidence will be discarded (TODO:?).

    Args:
        p_scenelet (str):
            Path to scenelet containing skeleton to interpolate.
        postfix (str):
            Tag to append to skeleton name before the json extension.
    Returns:
        o_path (str): Path, where it saved the interpolated version.

    """

    # Load skeleton
    assert os.path.exists(p_scenelet), "Does not exist: %s" % p_scenelet
    scenelet = Scenelet.load(p_scenelet)

    # Prepare animation time
    bpy.context.scene.render.fps = 10
    bpy.context.scene.render.fps_base = 1

    # Call rig importer
    import_skeleton_animation(
      skeleton=scenelet.skeleton,
      name='Output',
      add_camera=False,
      add_trajectory=False,
      time_scale=0.1,
      skeleton_transparency=0,
    )

    # Move head back to top (2D marker is at tip of head, but visually,
    # it looks better, if it's rendered with the center of the head around
    # the nose)
    for obj in bpy.data.objects:
        if obj.name.endswith("HEAD.Sphere"):
            cnstr = next(c for c in obj.constraints
                         if c.type == 'COPY_LOCATION')
            cnstr.head_tail = 1.

    # Extract Sphere endpoints as the new skeleton positions
    scenelet.skeleton = extract_skeleton(scene=bpy.context.scene)

    # Save to disk
    stem, ext = os.path.splitext(p_scenelet)
    o_path = "%s_%s.json" % (stem, postfix)
    scenelet.save(o_path, save_obj=True)

    # Return path
    return o_path


if __name__ == '__main__':
    args = parse_args(sys.argv)
    ipol_ik(p_scenelet=args.path, postfix=args.postfix)
    bpy.ops.wm.quit_blender()
