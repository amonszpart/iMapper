import sys
import pdb
import bdb
import traceback
import os
import math
import numpy as np
import argparse
try:
    import bpy
    from mathutils import Matrix, Vector, Quaternion, Euler
except ImportError:
    pass

from math import radians

# if the file is called directly (just for testing), add '../..' (imapper) to the python path
if __name__ == '__main__':
    p_imapper = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
    if p_imapper not in sys.path:
        sys.path.append(p_imapper)

print("realpath: {}".format(os.path.realpath(__file__)))
print(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.logic.joints import Joint
from imapper.logic.colors import stealth_colors
from imapper.util.json import json
from imapper.input.intrinsics import intrinsics_matrix


traj_colors = [
    '#a50606', '#984ea3', '#ff7f00',
    '#a65628', '#4daf4a', '#377eb8', '#444444', '#666666', '#aaaa33']
traj_colors = [
    tuple(int(c[1+i*2:1+i*2+2], 16) / 255.
          for i in range(3))
    for c in traj_colors
]

bone_color = (251/255, 154/255, 153/255)
JOINT_COLORS = {
    Joint.RELB.name: (255/255, 132/255, 0/255), # orange
    Joint.LELB.name: (255/255, 166/255, 71/255), # light orange
    Joint.RKNE.name: (152/255, 78/255, 163/255), # purple
    Joint.LKNE.name: (192/255, 145/255, 199/255), # light purple
    Joint.RANK.name: (152/255, 78/255, 163/255), # purple
    Joint.LANK.name: (192/255, 145/255, 199/255), # light purple
    Joint.LWRI.name: (255/255, 166/255, 71/255), # light orange
    Joint.RWRI.name: (255/255, 132/255, 0/255), # orange
    Joint.RHIP.name: (152/255, 78/255, 163/255), # purple
    Joint.LHIP.name: (152/255, 78/255, 163/255), # purple
    Joint.RSHO.name: (255/255, 132/255, 0/255), # orange
    Joint.LSHO.name: (255/255, 132/255, 0/255), # orange
    Joint.THRX.name: (255/255, 132/255, 0/255), # orange
    Joint.NECK.name: bone_color, # light red
    Joint.HEAD.name: (227/255, 26/255, 28/255), # red
    Joint.PELV.name: (152/255, 78/255, 163/255), # purple
}


def load_initial_scenelet(p_root, scene_name):
    p_file = os.path.normpath(
        os.path.join(p_root,
                     'skel_{}_unannot.json'.format(scene_name))
    )
    if not os.path.isfile(p_file):
        p_file = os.path.normpath(
            os.path.join(p_root, os.pardir,
                         'skel_{}_unannot.json'.format(scene_name))
        )
    sclt_input = Scenelet.load(p_file)

    return sclt_input


# to be called from blender
def import_scene(scene_filepath, video_filepath, localpose_filepath=None,
                 keypoint_filepath=None, intrinsics_filepath=None,
                 recording_res=None, render_width=None, cam_height=None,
                 clear_scene=True, time_scale=1, fps=24,
                 include_objects=True, include_skeleton=True,
                 include_video=True, include_keypoints=True,
                 include_localpose=True, sphere_scale=0.5):

    if render_width is None:
        render_width = 1920

    if clear_scene:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=True)

    # scene_filepath = os.path.join(scene_filepath, 'output', 'skel_output.json')

    bpy.context.scene.render.filepath = "%s/" % os.path.abspath(os.path.join(os.path.dirname(scene_filepath), 'render'))
    bpy.context.scene.render.fps = fps # the movie has to match this!
    bpy.context.scene.render.fps_base = 1 # actual fps is fps / fps_base
    if use_cycles:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.film_transparent = True

    # add light
    lamp = bpy.data.lamps.new(name="Lamp", type='SUN')
    # lamp = bpy.data.lamps.new(name="Lamp", type='AREA')
    lamp.color = (1.0, 1.0, 1.0)
    # lamp.shape = 'SQUARE'
    # lamp.size = 3.0
    lamp.shadow_soft_size = 1.0
    lamp.sky.use_sky = True
    if use_cycles:
        lamp.cycles.cast_shadow = True
        lamp.cycles.use_multiple_importance_sampling = True
        lamp.cycles.max_bounces = 1024
        lamp.use_nodes = True
        lamp.node_tree.nodes['Emission'].inputs[1].default_value = 3
    lamp_obj = bpy.data.objects.new(name="Lamp", object_data=lamp)
    bpy.context.scene.objects.link(lamp_obj)
    lamp_obj.location = (0., 3.3, 4.5)

    # import camera
    # intrinsics_filepath = os.path.join(os.path.dirname(scene_filepath), '../intrinsics.json')
    # recording_image = bpy.data.images.load(os.path.join(video_filepath, 'color_00001.jpg'))
    if video_filepath is not None and os.path.exists(video_filepath):
        recording_mclip = bpy.data.movieclips.load(video_filepath)
        recording_res = (recording_mclip.size[0], recording_mclip.size[1])
        bpy.data.movieclips.remove(recording_mclip)
    else:
        print('****** WARNING! Could not find the input video, using output size for intrinsics')
        recording_res = (render_width, (render_width) * (9./16))

    _ = import_camera(intrinsics_filepath=intrinsics_filepath, recording_res=recording_res, render_width=render_width)

    # load scenelet
    scenelet = Scenelet.load(scene_filepath)
    _p_root = os.path.normpath(
        os.path.abspath(os.path.dirname(intrinsics_filepath)))
    sclt_input = load_initial_scenelet(_p_root, scene_name)

    # import objects
    if include_objects:
        _ = import_objects(scenelet=scenelet)

    # import skeleton animation
    if include_skeleton:
        n_actors = scenelet.skeleton.n_actors
        if n_actors == 1:
            n_actors = max(n_actors, sclt_input.skeleton.n_actors)

        for actor_id in range(n_actors):
            import_skeleton_animation(
              skeleton=scenelet.skeleton,
              name='Output',
              add_camera=True,
              add_trajectory=True,
              time_scale=time_scale,
              skeleton_transparency=0,
              actor_id=actor_id,
              rate=sclt_input.aux_info['path_opt_params']['rate'],
              sphere_scale=sphere_scale
            )

    if include_localpose:
        attach_targets = {}
        attach_group = bpy.data.objects['Output.Actor.Group']
        joints = Joint.get_ordered_range()
        for joint in joints:
            joint_name = joint.get_name()
            if joint_name != 'PELV':
                attach_targets[joint_name] = \
                    bpy.data.objects['Output.' + joint_name + '.Target']
        localpose_skeleton = Scenelet.load(localpose_filepath).skeleton
        import_skeleton_animation(
            skeleton=localpose_skeleton,
            name='Localpose',
            add_camera=True,
            add_trajectory=False,
            time_scale=0.1,  # time given incorrectly as frame indices of a 10 fps clip
            location_offset=(0, -1, 0.85),
            attach_targets=attach_targets,
            attach_group=attach_group)

    if False:
        p_video_recs = os.path.normpath(
            os.path.join(os.path.dirname(scene_filepath),
                         os.pardir, os.pardir))
        p_fbx = os.path.join(p_video_recs, 'claire.fbx')
        bpy.ops.import_scene.fbx(filepath=p_fbx, use_anim=False,
                                 automatic_bone_orientation=True)
        attach_group = bpy.data.objects['Output.Actor.Group']
        arm_ = bpy.data.objects['Armature']
        arm_.parent = attach_group
    print('done importing animation')

    # create floor if necessary
    if not any('floor' in o.name for o in bpy.context.scene.objects):
        cam_height_ = cam_height
        cam_rot_ = None
        if cam_height_ is None:
            cam_height_ = -sclt_input.aux_info['ground'][1][3]
            cam_rot_ = [-cr for cr in sclt_input.aux_info['ground_rot']]
        # cam_height_ = cam_height if cam_height is not None \
        #     else scenelet.aux_info['ground'][1, 3]
        create_floor(cam_height=cam_height_, cam_rot=cam_rot_)

    # load keypoints
    if include_keypoints:
        import_keypoints(
            skeleton=scenelet.skeleton,
            keypoint_filepath=keypoint_filepath,
            intrinsics_filepath=intrinsics_filepath,
            recording_res=recording_res,
            time_scale=time_scale)

    # import video
    if include_video:
        if len(video_filepath) == 0:
            raise ValueError('need video directory path to import video')
        elif os.path.isfile(video_filepath):
            _ = import_video(video_filepath=video_filepath)
        else:
            print("Can't find video: %s" % video_filepath)

    print('done importing video')

    bpy.context.scene.update()
    bpy.data.materials.update()


def create_polyline(name, pts):
    curvedata = bpy.data.curves.new(name=name+'.Curve', type='CURVE')
    curvedata.dimensions = '3D'

    objectdata = bpy.data.objects.new(name+'.Object', curvedata)
    objectdata.location = (0, 0, 0) #object origin
    bpy.context.scene.objects.link(objectdata)

    polyline = curvedata.splines.new('POLY')
    polyline.points.add(len(pts)-1)

    for num, pt in enumerate(pts):
        polyline.points[num].co = (pt[0], pt[1], pt[2], 1)
        # polyline.points[num].co = (pt[0], pt[1], 0, 1)

    polyline.order_u = len(polyline.points)-1
    polyline.use_endpoint_u = True
    polyline.use_cyclic_u = False

    return objectdata, curvedata


def create_floor(cam_height=None, cam_rot=None):
    cam_height_ = cam_height
    if cam_height_ is None:
        cam_height = min(o.location.z - o.dimensions.z/2. for o in bpy.context.scene.objects)
    _ = bpy.ops.mesh.primitive_cube_add()
    floor_obj = bpy.context.selected_objects[-1]
    floor_obj.name = 'floor'
    floor_obj.location = (0, 0, cam_height_-0.05 if cam_height is None
                          else cam_height_)
    if cam_rot is not None:
        for c in range(3):
            floor_obj.rotation_euler[c] = np.deg2rad(cam_rot[c])
    floor_obj.dimensions = (30, 30, 0.1)
    grd_mat = create_diffuse_transparent_ao_material(material_name='Ground', color=(1, 1, 1))
    if len(floor_obj.data.materials) == 0:
        floor_obj.data.materials.append(grd_mat)
    else:
        floor_obj.data.materials[0] = grd_mat

    return floor_obj


def import_camera(intrinsics_filepath, recording_res, render_width):
    cam = bpy.data.cameras.new("Camera.Input")
    cam.sensor_fit = 'HORIZONTAL'
    obj = bpy.data.objects.new("Camera.Input", cam)
    obj.rotation_euler = (np.pi/2, 0., 0.)
    bpy.context.scene.objects.link(obj)

    render = bpy.context.scene.render
    render.resolution_x = render_width
    render.resolution_y = render_width * (recording_res[1] / recording_res[0])
    render.resolution_percentage = 100

    if intrinsics_filepath is not None:

        intrinsic_mat = json.load(open(intrinsics_filepath, 'r'))
        # intrinsic_mat = intrinsics_matrix(render_size[1], [render_size[1], render_size[0]], camera_type)

        f_in_mm = cam.lens
        fx = intrinsic_mat[0][0]
        fy = intrinsic_mat[1][1]

        # (pixel x / pixel y)
        pixel_aspect_ratio = fy / fx
        render.pixel_aspect_x = 1.
        render.pixel_aspect_y = 1 / pixel_aspect_ratio
        # if fx > fy:
        #     # print("fx %f > %f fy" % (fx, fy))
        #     render.pixel_aspect_x = 1.
        #     render.pixel_aspect_y = 1 / pixel_aspect_ratio
        # else:
        #     # print("fx %f < %f fy" % (fx, fy))
        #     render.pixel_aspect_x = fy / fx
        #     render.pixel_aspect_y = 1.
        # pixel_aspect_ratio = render.pixel_aspect_x / render.pixel_aspect_y
        # resolution_x_in_px = render.resolution_x
        # resolution_y_in_px = render.resolution_y
        # scale = render.resolution_percentage / 100

        # import pdb; pdb.set_trace()

        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio

        # fx / f_in_mm = resolution_x_in_px * scale / sensor_width_in_mm
        # sensor_width_in_mm = resolution_x_in_px * scale / fx * f_in_mm
        cam.sensor_width = (recording_res[0] / fx) * f_in_mm

        # # fy / f_in_mm = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
        # # sensor_height_in_mm = resolution_y_in_px * scale * pixel_aspect_ratio / fy * f_in_mm
        # cam.sensor_height = ((recording_res[1] / pixel_aspect_ratio) / fy) * f_in_mm

        maxdim = max(render.resolution_x, render.resolution_y)
        cam.shift_x = (render.resolution_x/2.0 - intrinsic_mat[0][2])/maxdim
        cam.shift_y = (intrinsic_mat[1][2] - render.resolution_y/2.0)/maxdim

    # add an ortographic camera
    cam2 = bpy.data.cameras.new("Camera.Viewing")
    obj2 = bpy.data.objects.new("Camera.Viewing", cam2)
    obj2.rotation_euler = (np.pi*(60/180), 0., 0.)
    obj2.location = (0., 0., 2.5)
    bpy.context.scene.objects.link(obj2)
    cam2.type = 'ORTHO'
    cam2.ortho_scale = 10

    return (obj, obj2)
    # print("shift set: %f,%f" % (cam.shift_x, cam.shift_y))

# def create_materials():

#     # create ground material
#     grd_mat = create_diffuse_transparent_ao_material(material_name='Ground', color=(1, 1, 1))

#     colors = stealth_colors
#     colors = colors[1::2] + colors[::2] # re-mix?
#     colors = [(c[0]/255., c[1]/255., c[2]/255.) for c in colors] # [0,255] to [0,1]

#     # create object materials
#     obj_mats = []
#     for i, color in enumerate(colors):
#         material_name = 'Object%04d' % i

#         mat = create_diffuse_transparent_ao_material(material_name=material_name, color=color)

#         obj_mats.append(mat)

#     # create actor materials
#     act_mats = []
#     for i, color in enumerate(colors):
#         material_name = 'Actor%04d' % i

#         mat = create_diffuse_transparent_ao_material(material_name=material_name, color=color)

#         act_mats.append(mat)

#     return obj_mats, act_mats, grd_mat


def create_diffuse_transparent_ao_material(material_name, color, ao=0,
                                           transparency=0, use_cycles=True):

    # remove existing material with the same name
    mat = next((m_ for m_ in bpy.data.materials if m_.name == material_name), None)
    if mat is not None:
        mat.user_clear()
        bpy.data.materials.remove(mat)
        mat = None

    # create new material
    mat = bpy.data.materials.new(name=material_name)
    mat.diffuse_color.r = color[0]
    mat.diffuse_color.g = color[1]
    mat.diffuse_color.b = color[2]
    if use_cycles:
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in list(nodes):
            nodes.remove(node)

        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.name = 'RGB'
        node_rgb.outputs[0].default_value = list(color) + [1]

        # mat.node_tree.nodes.active = node_rgb

        node_ao = nodes.new(type='ShaderNodeAmbientOcclusion')
        node_ao.name = 'AmbientOcclusion'
        node_ao.inputs[0].default_value = [0, 0, 0, 1.]

        # mat.node_tree.nodes.active = node_ao

        node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        node_diffuse.name = 'Diffuse'
        links.new(node_rgb.outputs[0], node_diffuse.inputs[0])

        # mat.node_tree.nodes.active = node_diffuse

        node_transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        node_transparent.name = 'Transparent'
        links.new(node_rgb.outputs[0], node_transparent.inputs[0])

        # mat.node_tree.nodes.active = node_transparent

        node_mix1 = nodes.new(type='ShaderNodeMixShader')
        node_mix1.name = 'Mix1'
        node_mix1.inputs[0].default_value = ao
        links.new(node_diffuse.outputs[0], node_mix1.inputs[1])
        links.new(node_ao.outputs[0], node_mix1.inputs[2])

        # mat.node_tree.nodes.active = node_mix1

        node_mix2 = nodes.new(type='ShaderNodeMixShader')
        node_mix2.name = 'Mix2'
        node_mix2.inputs[0].default_value = transparency
        links.new(node_mix1.outputs[0], node_mix2.inputs[1])
        links.new(node_transparent.outputs[0], node_mix2.inputs[2])

        # mat.node_tree.nodes.active = node_mix2

        # create output node
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.name = 'Output'
        links.new(node_mix2.outputs[0], node_output.inputs[0])

        mat.node_tree.update_tag()
        mat.node_tree.nodes.active = node_output

    return mat


def create_emissive_transparent_ao_material(material_name, color, ao=0, transparency=0):
    # remove existing material with the same name
    mat = next((m_ for m_ in bpy.data.materials if m_.name == material_name), None)
    if mat is not None:
        mat.user_clear()
        bpy.data.materials.remove(mat)
        mat = None

    # create new material
    mat = bpy.data.materials.new(name=material_name)
    mat.diffuse_color.r = color[0]
    mat.diffuse_color.g = color[1]
    mat.diffuse_color.b = color[2]
    if use_cycles:
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in list(nodes):
            nodes.remove(node)

        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.name = 'RGB'
        node_rgb.outputs[0].default_value = list(color) + [1]

        node_ao = nodes.new(type='ShaderNodeAmbientOcclusion')
        node_ao.name = 'AmbientOcclusion'
        node_ao.inputs[0].default_value = [0, 0, 0, 1.]

        node_emission = nodes.new(type='ShaderNodeEmission')
        node_emission.name = 'Emission'
        links.new(node_rgb.outputs[0], node_emission.inputs[0])

        node_transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        node_transparent.name = 'Transparent'
        links.new(node_rgb.outputs[0], node_transparent.inputs[0])

        node_mix1 = nodes.new(type='ShaderNodeMixShader')
        node_mix1.name = 'Mix1'
        node_mix1.inputs[0].default_value = ao
        links.new(node_emission.outputs[0], node_mix1.inputs[1])
        links.new(node_ao.outputs[0], node_mix1.inputs[2])

        node_mix2 = nodes.new(type='ShaderNodeMixShader')
        node_mix2.name = 'Mix2'
        node_mix2.inputs[0].default_value = transparency
        links.new(node_mix1.outputs[0], node_mix2.inputs[1])
        links.new(node_transparent.outputs[0], node_mix2.inputs[2])

        # create output node
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.name = 'Output'
        links.new(node_mix2.outputs[0], node_output.inputs[0])

        mat.node_tree.update_tag()

        mat.node_tree.nodes.active = node_output

    return mat

def import_objects(scenelet):

    # obj_filepaths = list(set(p.path for o in  for p in o.parts.values()))
    obj_import = [o for g in scenelet.objects.values() for o in g.parts.values()]
    obj_labels = [g.label for g in scenelet.objects.values() for o in g.parts.values()]

    # obj_labels = list(set(o for g in scenelet.objects.values() for o in g.parts.values()))
    obj_filepaths = [o.path for o in obj_import]
    # obj_labels = [o.label for o in objs]

    # label_colors = {
    #     'bed': (141/255, 211/255, 199/255),
    #     'chair': (253/255, 180/255, 98/255),
    #     'couch': (179/255, 222/255, 105/255),
    #     'monitor': (252/255, 205/255, 229/255),
    #     'plant': (252/255, 205/255, 229/255),
    #     'shelf': (255/255, 255/255, 179/255),
    #     'table': (255/255, 237/255, 111/255),
    #     'tv': (252/255, 205/255, 229/255),
    #     'whiteboard': (204/255, 235/255, 197/255),
    #     }

    label_colors = {
        'bed': (253/255, 191/255, 111/255), # light orange
        'chair': (61/255, 153/255, 112/255), # olive green # (255/255, 127/255, 0/255), # orange
        'couch': (51/255, 160/255, 44/255), # green
        'monitor': (126/255, 180/255, 213/255), # light blue
        'plant': (178/255, 223/255, 138/255), # light green
        'shelf': (255/255, 237/255, 111/255), # yellow
        'table': (31/255, 120/255, 180/255), # blue
        'tv': (251/255, 154/255, 153/255), # light red
        'whiteboard': (253/255, 244/255, 182/255), # light yellow
        }

    #     Joint.LANK.name: (126/255, 180/255, 213/255), # light blue
    #     Joint.LWRI.name: (216/255, 165/255, 209/255), # light purple
    #     Joint.LHIP.name: (31/255, 120/255, 180/255), # blue
    #     Joint.RSHO.name: (152/255, 78/255, 163/255), # purple

    object_materials = {}
    for lbl, clr in label_colors.items():
        object_materials[lbl] = create_diffuse_transparent_ao_material(material_name='Object.'+lbl, color=clr)
    ground_material = create_diffuse_transparent_ao_material(material_name='Ground', color=(1, 1, 1))

    group = bpy.data.objects.new("Objects.Group", None)
    group.empty_draw_size = 0.1
    bpy.context.scene.objects.link(group)

    prefix = 'Object'
    objs = []
    for i, _ in enumerate(obj_import):
        obj_filepath = obj_filepaths[i]
        obj_label = obj_labels[i]

        if os.path.splitext(obj_filepath)[1] == '.obj':
            obj_filepath = os.path.splitext(obj_filepath)[0] + '.json'

        # load data from json file
        obj_name = os.path.splitext(os.path.basename(obj_filepath))[0]
        with open(obj_filepath, 'r') as f:
            data = json.load(f)
        location = np.squeeze(np.array(data['centroid']))
        scale = np.squeeze(np.array(data['scales']))
        axes = np.array(data['axes'])
        part_id = data['part_id'] if 'part_id' in data else -1

        # world transform matrix
        location = np.array(stealth2blender_coords(location))
        axes[:, 0] = stealth2blender_coords(axes[:, 0])
        axes[:, 1] = stealth2blender_coords(axes[:, 1])
        axes[:, 2] = stealth2blender_coords(axes[:, 2])
        scale = [abs(s) for s in scale]
        mat = Matrix.Translation((location[0], location[1], location[2]))
        scale = Matrix([
            [scale[0] / 2., 0., 0.],
            [0., scale[1] / 2., 0.],
            [0., 0., scale[2] / 2.]]).to_4x4()
        rot = Matrix([
            axes[0, :].tolist(),
            axes[1, :].tolist(),
            axes[2, :].tolist()]).to_4x4()
        mat = mat * rot * scale

        # add to blender
        _ = bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.selected_objects[-1]
        obj.matrix_world = mat
        obj.name = "%s.%s" % (prefix, obj_name)
        obj['part_id'] = part_id
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
        # obj['obj_id'] = int(obj_filepath.split('_')[0])

        # add material
        mat = ground_material if 'floor' in obj.name else object_materials[obj_label]
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        obj.parent = group

        objs.append(obj)

    return objs


def get_blender_frame_time(skeleton, frame_id, rate, time_scale, actor_id):
    """Goes from multi-actor integer frame_id to modded blender float time."""
    # stays within video frame limits
    frame_id2 = skeleton.mod_frame_id(frame_id=frame_id)  # type: int
    time_ = skeleton.get_time(frame_id)
    if actor_id > 0:
        time_ = frame_id2 / rate
    print('time is {} for {} ({}), orig time: {}, rate: {}, '
          'time_scale: {}'
          .format(time_, frame_id, frame_id2,
                  skeleton.get_time(frame_id), rate, time_scale))
    frame_time = time_ * time_scale
    return frame_time


def import_skeleton_animation(skeleton, name, add_camera, add_trajectory,
                              time_scale=1, location_offset=None,
                              skeleton_transparency=0, attach_targets=None,
                              attach_group=None, actor_id=0,
                              rate=None, sphere_scale=1.):
    context = bpy.context
    unconstrained_joints = ('NECK')

    if location_offset is None:
        location_offset = (0, 0, 0)

    max_time = 0
    if actor_id == 0:
        context.scene.frame_end = 0
    # max_time = max(skeleton.get_time(frame_id)
    #                for frame_id, _ in skeleton._frame_ids.items()) * time_scale
    # context.scene.frame_end = max(context.scene.frame_end, max_time*context.scene.render.fps)

    print('*********** time:')
    # print(max_time)
    postfix = '' if actor_id == 0 else '.actor{:02d}'.format(actor_id)
    group = bpy.data.objects.new('{}.Actor.Group{}'.format(name, postfix),
                                 None)
    group.empty_draw_size = 0.1
    context.scene.objects.link(group)

    joints = Joint.get_ordered_range()

    bone_radius = 0.04
    material_name = name+'.Bone.Material'
    if actor_id == 0 or material_name not in bpy.data.materials:
        bone_material = create_diffuse_transparent_ao_material(
          material_name=material_name, color=bone_color,
          transparency=skeleton_transparency)
    else:
        bone_material = bpy.data.materials.get(material_name)

    # get median length of joints over the animation
    joint_len = {}
    # joint_color = {}
    for joint in joints:
        parent = joint.get_parent()
        joint_name = joint.get_name()
        if parent is not None:
            joint_len[joint_name] = np.median(
              np.linalg.norm(
                skeleton.poses[:, :, joint] - skeleton.poses[:, :, parent],
                ord=2, axis=1), axis=0)
        else:
            joint_len[joint_name] = 0

    # Aron on 6/4/2018:
    # scale neck joint smaller and head joint longer
    tmp = joint_len[Joint.NECK.get_name()]
    joint_len[Joint.NECK.get_name()] = tmp * 0.5
    joint_len[Joint.HEAD.get_name()] += tmp * 0.5
    # average sides:
    for j0, j1 in (('LHIP', 'RHIP'), ('LKNE', 'RKNE'), ('LANK', 'RANK'),
                   ('LSHO', 'RSHO'), ('LELB', 'RELB'), ('LWRI', 'RWRI')):
        avg = (joint_len[j0] + joint_len[j1]) / 2.
        joint_len[j0] = avg
        joint_len[j1] = avg
    # print(joint_len)

    # joint_color = {
    #     Joint.RELB.name: (31/255, 120/255, 180/255), # blue
    #     Joint.LELB.name: (152/255, 78/255, 163/255), # purple  (253/255, 192/255, 134/255), # yellow  (51/255, 160/255, 44/255), # green
    #     Joint.RKNE.name: (166/255, 206/255, 227/255), # light blue
    #     Joint.LKNE.name: (244/255, 202/255, 228/255), # light purple   (230/255, 245/255, 201/255), # light yellow   (178/255, 223/255, 138/255), # light green
    #     Joint.RANK.name: (166/255, 206/255, 227/255), # light blue
    #     Joint.LANK.name: (244/255, 202/255, 228/255), # light purple   (230/255, 245/255, 201/255), # light yellow   (178/255, 223/255, 138/255), # light green
    #     Joint.LWRI.name: (152/255, 78/255, 163/255), # purple  (253/255, 192/255, 134/255), # yellow  (51/255, 160/255, 44/255), # green
    #     Joint.RWRI.name: (31/255, 120/255, 180/255), # blue
    #     Joint.RHIP.name: (166/255, 206/255, 227/255), # light blue
    #     Joint.LHIP.name: (244/255, 202/255, 228/255), # light purple   (230/255, 245/255, 201/255), # light yellow   (178/255, 223/255, 138/255), # light green
    #     Joint.RSHO.name: (31/255, 120/255, 180/255), # blue
    #     Joint.LSHO.name: (152/255, 78/255, 163/255), # purple  (253/255, 192/255, 134/255), # yellow  (51/255, 160/255, 44/255), # green
    #     Joint.THRX.name: bone_color, # light red
    #     Joint.NECK.name: bone_color, # light red
    #     Joint.HEAD.name: (227/255, 26/255, 28/255), # red
    #     Joint.PELV.name: bone_color # light red
    # }

    # joint_color = {
    #     Joint.RELB.name: (152/255, 78/255, 163/255), # purple
    #     Joint.LELB.name: (216/255, 165/255, 209/255), # light purple # (244/255, 202/255, 228/255), # light purple old
    #     Joint.RKNE.name: (31/255, 120/255, 180/255), # blue
    #     Joint.LKNE.name: (126/255, 180/255, 213/255), # light blue
    #     Joint.RANK.name: (31/255, 120/255, 180/255), # blue
    #     Joint.LANK.name: (126/255, 180/255, 213/255), # light blue # (166/255, 206/255, 227/255), # light blue old
    #     Joint.LWRI.name: (216/255, 165/255, 209/255), # light purple
    #     Joint.RWRI.name: (152/255, 78/255, 163/255), # purple
    #     Joint.RHIP.name: (31/255, 120/255, 180/255), # blue
    #     Joint.LHIP.name: (31/255, 120/255, 180/255), # blue
    #     Joint.RSHO.name: (152/255, 78/255, 163/255), # purple
    #     Joint.LSHO.name: (152/255, 78/255, 163/255), # purple
    #     Joint.THRX.name: (152/255, 78/255, 163/255), # purple
    #     Joint.NECK.name: bone_color, # light red
    #     Joint.HEAD.name: (227/255, 26/255, 28/255), # red
    #     Joint.PELV.name: (31/255, 120/255, 180/255), # blue
    # }

    # joint_color = {
    #     Joint.RELB.name: (255/255, 133/255, 27/255), # orange
    #     Joint.LELB.name: (255/255, 174/255, 104/255), # light orange
    #     Joint.RKNE.name: (255/255, 220/255, 0/255), # yellow
    #     Joint.LKNE.name: (255/255, 238/255, 130/255), # light yellow
    #     Joint.RANK.name: (255/255, 220/255, 0/255), # yellow
    #     Joint.LANK.name: (255/255, 238/255, 130/255), # light yellow
    #     Joint.LWRI.name: (255/255, 174/255, 104/255), # light orange
    #     Joint.RWRI.name: (255/255, 133/255, 27/255), # orange
    #     Joint.RHIP.name: (255/255, 220/255, 0/255), # yellow
    #     Joint.LHIP.name: (255/255, 220/255, 0/255), # yellow
    #     Joint.RSHO.name: (255/255, 133/255, 27/255), # orange
    #     Joint.LSHO.name: (255/255, 133/255, 27/255), # orange
    #     Joint.THRX.name: (255/255, 133/255, 27/255), # orange
    #     Joint.NECK.name: bone_color,
    #     Joint.HEAD.name: (227/255, 26/255, 28/255), # red
    #     Joint.PELV.name: (255/255,220/255,0/255), # yellow
    # }

    # joint_color = {
    #     Joint.RELB.name: (152/255, 78/255, 163/255), # purple
    #     Joint.LELB.name: (216/255, 165/255, 209/255), # light purple
    #     Joint.RKNE.name: (255/255, 156/255, 0/255), # orange
    #     Joint.LKNE.name: (255/255, 189/255, 84/255), # light orange
    #     Joint.RANK.name: (255/255, 156/255, 0/255), # orange
    #     Joint.LANK.name: (255/255, 189/255, 84/255), # light orange
    #     Joint.LWRI.name: (216/255, 165/255, 209/255), # light purple
    #     Joint.RWRI.name: (152/255, 78/255, 163/255), # purple
    #     Joint.RHIP.name: (255/255, 156/255, 0/255), # orange
    #     Joint.LHIP.name: (255/255, 156/255, 0/255), # orange
    #     Joint.RSHO.name: (152/255, 78/255, 163/255), # purple
    #     Joint.LSHO.name: (152/255, 78/255, 163/255), # purple
    #     Joint.THRX.name: (152/255, 78/255, 163/255), # purple
    #     Joint.NECK.name: bone_color, # light red
    #     Joint.HEAD.name: (227/255, 26/255, 28/255), # red
    #     Joint.PELV.name: (255/255, 156/255, 0/255), # orange
    # }


    joint_tail_size = {
        Joint.RELB.name: 0.08,
        Joint.LELB.name: 0.08,
        Joint.RKNE.name: 0.08,
        Joint.LKNE.name: 0.08,
        Joint.RANK.name: 0.08,
        Joint.LANK.name: 0.08,
        Joint.LWRI.name: 0.08,
        Joint.RWRI.name: 0.08,
        Joint.RHIP.name: 0.08,
        Joint.LHIP.name: 0.08,
        Joint.RSHO.name: 0.08,
        Joint.LSHO.name: 0.08,
        Joint.THRX.name: 0.08,
        Joint.NECK.name: 0.02,
        Joint.HEAD.name: 0.14,
        Joint.PELV.name: 0.08
    }

    rig_name = '{}.Rig{}'.format(name, postfix)
    rig = import_rig(name=rig_name, joints=joints, joint_len=joint_len)
    rig.pose.ik_param.reiteration_method = 'NEVER'
    rig.parent = group

    for joint in joints:

        joint_name = joint.get_name()

        # pelvis joint is not a bone (it is the local coordinate frame
        # origin of the rig/object)
        if joint_name != 'PELV':

            bone = rig.pose.bones[rig_name + '.' + joint_name]

            if joint_name not in unconstrained_joints:
                # joint target object
                # target = bpy.data.objects.new(name+'.'+joint_name+'.Target', None)
                target = bpy.data.objects.new(
                  '{}.{}.Target{}'.format(name, joint_name, postfix), None)
                target.empty_draw_size = 0.1
                context.scene.objects.link(target)

                # add distance to joint target as constraint
                constr = bone.constraints.new(type='IK')
                constr.target = target
                constr.use_tail = True
                constr.use_stretch = False
                constr.ik_type = 'DISTANCE'
                # constr.iterations = 500
                constr.chain_count = 0
                constr.distance = 0.01

                target.parent = group

            # add cylinder for every bone
            bpy.ops.mesh.primitive_cylinder_add(
                radius=bone_radius * sphere_scale,
                depth=joint_len[joint_name]
            )
            bone_cyl = context.selected_objects[-1]
            bone_cyl.name = '{}.{}.Cylinder{}' \
                .format(name, joint_name, postfix)
            bone_cyl.rotation_euler[0] = radians(90.)
            bpy.ops.object.transform_apply(rotation=True)

            #
            # Cylinder location:
            #
            # v0 (Paul):
            # bone_cyl.location = (bone.head + bone.tail) / 2
            #
            # v1 (Aron):
            bpy.ops.object.constraint_add(type='COPY_LOCATION')
            cnstr = next(c for c in context.object.constraints
                         if c.type == 'COPY_LOCATION')
            cnstr.target = rig
            cnstr.subtarget = bone.name
            cnstr.head_tail = 0.5

            #
            # Cylinder rotation:
            #
            # v0 (Paul):
            # bone_cyl.rotation_mode = 'QUATERNION'
            # new_dir = (bone.head - bone.tail).normalized()
            # new_dir.z = -new_dir.z
            # old_dir = Vector([0, 0, -1])
            # axis = new_dir.cross(old_dir).normalized()
            # angle = new_dir.angle(old_dir)
            # bone_cyl.rotation_quaternion = Quaternion(axis, angle)
            # v1 (Aron):
            bpy.ops.object.constraint_add(type='COPY_ROTATION')
            cnstr = next(c for c in context.object.constraints
                         if c.type == 'COPY_ROTATION')
            cnstr.target = rig
            cnstr.subtarget = bone.name
            # cnstr.head_tail = 0.5

            #
            # Cylinder material:
            #

            if len(bone_cyl.data.materials) == 0:
                bone_cyl.data.materials.append(bone_material)
            else:
                bone_cyl.data.materials[0] = bone_material

            bone_cyl.parent = group

        # add sphere for every joint
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=4,
            size=joint_tail_size[joint_name] * sphere_scale)
        bone_sph = context.selected_objects[-1]
        bone_sph.name = '{}.{}.Sphere{}'.format(name, joint_name, postfix)
        # name+'.'+joint_name+'.Sphere'

        # Sphere location:
        # v0 (Paul):
        # bone_sph.rotation_mode = 'QUATERNION'
        # if joint_name == 'HEAD':
        #     bone = rig.pose.bones[rig_name+'.'+joint_name]
        #     bone_sph.location = bone.tail - Vector([0, 0, 0.05])
        # elif joint_name == 'PELV':
        #     bone = rig.pose.bones[rig_name+'.THRX']
        #     bone_sph.location = bone.head
        # else:
        #     bone = rig.pose.bones[rig_name+'.'+joint_name]
        #     bone_sph.location = bone.tail
        bone_name = "%s.%s" % (rig_name, joint_name) \
            if joint_name != 'PELV' else "%s.THRX" % rig_name
        bone = rig.pose.bones[bone_name]

        # v1 (Aron):
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        cnstr = next(c for c in context.object.constraints
                     if c.type == 'COPY_LOCATION')
        cnstr.target = rig
        cnstr.subtarget = bone.name
        cnstr.head_tail = 0. if joint_name == 'PELV' \
            else 0.75 if joint_name == 'HEAD' \
            else 1.

        sph_mat_name = name + '.' + joint_name + '.Material'
        sph_mat = bpy.data.materials.get(sph_mat_name)
        if sph_mat is None:
            sph_mat = create_diffuse_transparent_ao_material(
              material_name=sph_mat_name, color=JOINT_COLORS[joint_name],
              transparency=skeleton_transparency)
        if len(bone_sph.data.materials) == 0:
            bone_sph.data.materials.append(sph_mat)
        else:
            bone_sph.data.materials[0] = sph_mat

        bone_sph.parent = group

    first_frame_id = skeleton.get_actor_first_frame(actor_id=actor_id)
    last_frame_id = skeleton.get_actor_last_frame(actor_id=actor_id)
    fps = context.scene.render.fps / context.scene.render.fps_base
    # set group and target locations
    for frame_id, frame_ind in sorted(skeleton._frame_ids.items()):
        if frame_id < first_frame_id or last_frame_id < frame_id:
            continue
        # if actor_id != skeleton.get_actor_id(frame_id):
        #     continue

        print('frame %d / %d' % (frame_id, len(skeleton._frame_ids)))
        frame_time = get_blender_frame_time(skeleton, frame_id, rate,
                                            time_scale, actor_id)
        max_time = max(max_time, frame_time)

        frame_ind_blender = frame_time * fps
        context.scene.frame_set(
          frame=int(frame_ind_blender),
          subframe=frame_ind_blender - int(frame_ind_blender))

        # set group location to pelvis (also moves targets)
        pelvis_location = Vector(stealth2blender_coords(
          skeleton.poses[frame_ind, :, Joint.PELV]))
        # group.location = location_from.location if location_from is not None else pelvis_location + Vector(location_offset)
        if attach_targets is not None and attach_group is not None:
            offset = Vector([0, 0, 0])
            joint_count = 0
            for joint in joints:
                joint_location = Vector(stealth2blender_coords(
                  skeleton.poses[frame_ind, :, joint])) - pelvis_location
                joint_name = joint.get_name()
                if joint_name != 'PELV':
                    offset += attach_targets[joint_name].location - joint_location
                    joint_count += 1
            offset /= joint_count
            group.location = offset + attach_group.location
        else:
            group.location = pelvis_location + Vector(location_offset)
        group.keyframe_insert(data_path='location')

        for joint in joints:
            joint_location = Vector(stealth2blender_coords(
              skeleton.poses[frame_ind, :, joint]))
            joint_name = joint.get_name()

            # set target location relative to pelvis
            if joint_name not in ('PELV', 'NECK'):
                target_name = '{}.{}.Target{}' \
                    .format(name, joint_name, postfix)
                # name+'.'+joint_name+'.Target'
                target = bpy.data.objects[target_name]
                target.location = joint_location - pelvis_location
                target.keyframe_insert(data_path='location')

                # bone = rig.pose.bones[rig_name+'.'+joint_name]

    bpy.ops.object.mode_set(mode='OBJECT')

    for ob in bpy.data.objects:
        if ob.parent is not None and ob.parent.name == group.name:
            for frame, h2 in [(first_frame_id, True), (last_frame_id, False)]:
                frame_time = get_blender_frame_time(skeleton, frame, rate,
                                                    time_scale, actor_id) * fps
                for dt, flip in [(-0.5, True), (0., False)]:
                    do_hide = flip
                    frame_time_ = frame_time + dt if h2 else frame_time - dt
                    ob.hide = do_hide
                    ob.keyframe_insert(data_path='hide', frame=frame_time_)
                    ob.hide_render = do_hide
                    ob.keyframe_insert(data_path='hide_render',
                                       frame=frame_time_)
                    if do_hide:
                        print('hiding {} at {}, ({}, {})'
                              .format(ob.name, frame_time_, first_frame_id,
                                      last_frame_id))
                    else:
                        print('showing {} at {}, ({}, {})'
                              .format(ob.name, frame_time_, first_frame_id,
                                      last_frame_id))
        # else:
        #     print('{}\'s parent: {}, group: {}, {}'
        #           .format(ob.name, ob.parent, group, group.name))
    # bpy.ops.object

    # add an ortographic camera in the local coordinate frame of the skeleton
    if add_camera:
        cam = bpy.data.cameras.new(name+'.Actor.Camera'+postfix)
        cam.type = 'ORTHO'
        cam.ortho_scale = 3.2
        obj = bpy.data.objects.new(name+'.Actor.Camera'+postfix, cam)
        obj.rotation_euler = (np.pi*(3/8), 0., 0.)
        obj.location = (0., -2.7, 1.)
        context.scene.objects.link(obj)
        obj.parent = group

    # add the pelvis trajectory
    if add_trajectory:
        traj_mat = create_diffuse_transparent_ao_material(
          material_name=name+'.Actor.Traj.Material'+postfix,
          color=traj_colors[actor_id % len(traj_colors)])

        for joint in joints:

            joint_name = joint.get_name()

            if joint_name != 'PELV':
                continue

            joint_location = []
            for frame_id, frame_ind in sorted(skeleton._frame_ids.items()):
                if skeleton.get_actor_id(frame_id=frame_id) == actor_id:
                    joint_location.append(
                      stealth2blender_coords(
                        skeleton.poses[frame_ind, :, joint]))

            # frame_ind_joint_location = np.concatenate(
            #   (
            #       np.expand_dims(np.array(list(skeleton._frame_ids.values())),-1),
            #       np.array(joint_location)
            #   ), axis=1)
            # import pdb; pdb.set_trace()
            # np.savetxt('../../../../submissions/Siggraph2018/image_sources/top_view_joint_locations.txt', frame_ind_joint_location)

            pl, pl_data = create_polyline(
              name=name+'.'+joint_name+'.Traj'+postfix, pts=joint_location)
            pl_data.bevel_depth = 0.012 if joint_name == 'PELV' else 0.004
            pl_data.bevel_resolution = 5
            pl_data.fill_mode = 'FULL'
            pl_data.twist_mode = 'TANGENT'

            if len(pl.data.materials) > 0:
                pl.data.materials[0] = traj_mat
            else:
                pl.data.materials.append(traj_mat)

            pl_smooth = pl.modifiers.new(
              name=name+'.'+joint_name+'.Traj.Smoother'+postfix, type='SMOOTH')
            pl_smooth.factor = 1
            pl_smooth.iterations = 12

    rig.pose.ik_param.reiteration_method = 'ALWAYS'

    context.scene.frame_end = max(context.scene.frame_end, max_time * fps)

    return pelvis_location


def import_rig(name, joints, joint_len):
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    obj = bpy.context.object
    # obj.rotation_euler[0] = math.pi
    # obj.show_x_ray = True
    obj.name = name
    amt = obj.data
    amt.name = obj.name + 'Amt'
    amt.show_axes = True
    amt.draw_type = 'STICK'

    bpy.ops.object.mode_set(mode='EDIT')

    for joint in joints:
        joint_name = joint.get_name()
        parent = joint.get_parent()

        bone = amt.edit_bones.new(name+'.'+joint_name)

        if parent is not None:
            parent_name = parent.get_name()
            parent = amt.edit_bones[name+'.'+parent_name]

            bone.parent = parent
            bone.head = parent.tail
            bone.use_connect = True
            (_, rot, _) = parent.matrix.decompose()
            bone_vec = Vector(stealth2blender_coords(joint.get_bone_from_parent()))
            bone_vec.z *= -1
            bone.tail = rot * (bone_vec * joint_len[joint.get_name()]) + bone.head
        else:
            bone.tail = bone.head

    bpy.ops.object.mode_set(mode='POSE')
    obj.pose.ik_solver = 'ITASC'
    obj.pose.ik_param.mode = 'SIMULATION'
    obj.pose.ik_param.precision = 0.1  # Paul's: 1e-4
    obj.pose.ik_param.iterations = 100
    obj.pose.ik_param.reiteration_method = 'ALWAYS'

    # for joint in [Joint.LHIP, Joint.RHIP]:
    #     bone = obj.pose.bones[joint.get_name()]
    #     bone.lock_ik_x = True
    #     bone.lock_ik_z = True

    for joint in [Joint.LSHO, Joint.RSHO]:
        bone = obj.pose.bones[name+'.'+joint.get_name()]
        # bone.lock_ik_z = True
        bone.ik_min_x = math.radians(-30)
        bone.ik_max_x = math.radians(30)
        bone.use_ik_limit_x = True

    for joint in (Joint.LANK, Joint.RANK, Joint.RKNE, Joint.LKNE):
        bone = obj.pose.bones[name + '.' + joint.get_name()]
        bone.lock_ik_y = True
    # for joint in (Joint.LHIP, Joint.RHIP):
    #     bone = obj.pose.bones[name + '.' + joint.get_name()]
    #     bone.lock_ik_x = True

    # lock NECK
    bone = obj.pose.bones[name + '.' + Joint.NECK.get_name()]
    bone.use_ik_limit_x = True
    bone.ik_min_x = math.radians(-20)
    bone.ik_max_x = math.radians(20)
    bone.use_ik_limit_y = True
    bone.ik_min_y = math.radians(-20)
    bone.ik_max_y = math.radians(20)
    bone.use_ik_limit_z = True
    bone.ik_min_z = math.radians(-20)
    bone.ik_max_z = math.radians(20)
    # bone.ik_stiffness_x = 0.8
    # bone.ik_stiffness_y = 0.8
    # bone.ik_stiffness_z = 0.8
    # bone.lock_ik_x = True
    # bone.lock_ik_y = True
    # bone.lock_ik_z = True

    bpy.ops.object.mode_set(mode='OBJECT')

    return obj


def import_keypoints(skeleton, keypoint_filepath, intrinsics_filepath, recording_res, time_scale=1):

    plane_depth = 1
    keypoint_size = 0.003
    denis_res_y = 368
    skeleton_2d_time_scaling = 0.1 # time given incorrectly as frame indices of a 10 fps clip

    skeleton_2d = Scenelet.load(keypoint_filepath, no_obj=True).skeleton

    joints = Joint.get_ordered_range()

    ikp_group = bpy.data.objects.new("Keypoint.Input.Group", None)
    ikp_group.empty_draw_size = 0.1
    bpy.context.scene.objects.link(ikp_group)

    okp_group = bpy.data.objects.new("Keypoint.Output.Group", None)
    okp_group.empty_draw_size = 0.1
    bpy.context.scene.objects.link(okp_group)

    offset_group = bpy.data.objects.new("Keypoint.Offset.Group", None)
    offset_group.empty_draw_size = 0.1
    bpy.context.scene.objects.link(offset_group)

    ikp_mat = create_emissive_transparent_ao_material(material_name='Keypoint.Input', color=(1, 0, 0))
    okp_mat = create_emissive_transparent_ao_material(material_name='Keypoint.Output', color=(0, 1, 0))
    offset_mat = create_emissive_transparent_ao_material(material_name='Keypoint.Offset', color=(0, 0, 0))

    # define keypoint objects
    for joint in joints:

        joint_name = joint.get_name()

        # create input keypoint object
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=4,
            size=keypoint_size)
        ikp = bpy.context.selected_objects[-1]
        # import pdb; pdb.set_trace()
        ikp.name = joint_name+'.Input.Keypoint'
        # import pdb; pdb.set_trace()
        ikp.parent = ikp_group
        if len(ikp.data.materials) > 0:
            ikp.data.materials[0] = ikp_mat
        else:
            ikp.data.materials.append(ikp_mat)

        # create output keypoint object
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=4,
            size=keypoint_size)
        okp = bpy.context.selected_objects[-1]
        okp.name = joint_name+'.Output.Keypoint'
        okp.parent = okp_group
        if len(okp.data.materials) > 0:
            okp.data.materials[0] = okp_mat
        else:
            okp.data.materials.append(okp_mat)

        # add cylinder for pair of points
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.002,
            depth=1,
        )
        offset_cyl = bpy.context.selected_objects[-1]
        offset_cyl.name = joint_name+'.Offset'
        offset_cyl.rotation_mode = 'QUATERNION'
        offset_cyl.location = ((okp.location + ikp.location) / 2) * 1.01
        offset_cyl.scale = (1, 1, (okp.location - ikp.location).magnitude * 1.01)
        # new_dir = (okp.location - ikp.location).normalized()
        # new_dir.z = -new_dir.z
        new_dir = Vector([0, 0, -1])
        old_dir = Vector([0, 0, -1])
        axis = new_dir.cross(old_dir).normalized()
        angle = new_dir.angle(old_dir)
        offset_cyl.rotation_quaternion = Quaternion(axis, angle)
        if len(offset_cyl.data.materials) == 0:
            offset_cyl.data.materials.append(offset_mat)
        else:
            offset_cyl.data.materials[0] = offset_mat
        offset_cyl.parent = offset_group

        # # create connecting line and hook to input/output keypoints
        # curve, curvedata = create_polyline(name=joint_name+'.Offset', pts=[tuple(ikp.location), tuple(okp.location)])
        # start_hook = curve.modifiers.new(joint_name+'.Offset.Starthook', 'HOOK')
        # start_hook.object = ikp
        # end_hook = curve.modifiers.new(joint_name+'.Offset.Endhook', 'HOOK')
        # end_hook.object = okp

        # curve.parent = group
        # curvedata.bevel_depth = 0.002
        # curvedata.bevel_resolution = 5
        # curvedata.fill_mode = 'FULL'
        # bpy.context.scene.objects.active = curve

        # bpy.ops.object.mode_set(mode='EDIT')
        # start_point = curvedata.splines[0].points[0]
        # end_point = curvedata.splines[0].points[1]

        # print(joint_name)

        # start_point.select = True
        # bpy.ops.object.hook_assign(modifier=joint_name+'.Offset.Starthook')
        # start_point.select = False

        # end_point.select = True
        # bpy.ops.object.hook_assign(modifier=joint_name+'.Offset.Endhook')
        # end_point.select = False

        # bpy.ops.object.mode_set(mode='OBJECT')

    intrinsic_mat = json.load(open(intrinsics_filepath, 'r'))
    fx = intrinsic_mat[0][0] # in pixels of recording res.
    fy = intrinsic_mat[1][1] # in pixels of recording res.
    cx = intrinsic_mat[0][2] # in pixels of recording res.
    cy = intrinsic_mat[1][2] # in pixels of recording res.

    # animate input keypoint objects
    for frame_id, frame_ind in skeleton_2d._frame_ids.items():
        frame_time = skeleton_2d.get_time(frame_id) * skeleton_2d_time_scaling
        frame_ind_blender = frame_time * (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
        bpy.context.scene.frame_set(frame=int(frame_ind_blender), subframe=frame_ind_blender - int(frame_ind_blender))

        for joint in joints:
            joint_name = joint.get_name()

            ikp_location = skeleton_2d.poses[frame_ind, :, joint]
            # from denis res to recording res
            ikp_location *= recording_res[1] / denis_res_y
            # from pixel coordinates to 3d coordinates at depth = plane_depth: x3d = ((x2d - cx) / fx) * z3d
            ikp_location[0] = ((ikp_location[0] - cx) / fx) * plane_depth
            ikp_location[1] = ((ikp_location[1] - cy) / fy) * plane_depth
            ikp_location[2] = plane_depth
            # to blender coordinates
            ikp_location = Vector(stealth2blender_coords(ikp_location))
            # animate blender object
            ikp = bpy.data.objects[joint_name+'.Input.Keypoint']
            ikp.location = ikp_location
            ikp.keyframe_insert(data_path='location')

    # animate output keypoint objects
    for frame_id, frame_ind in skeleton._frame_ids.items():
        frame_time = skeleton.get_time(frame_id) * time_scale
        frame_ind_blender = frame_time * (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
        bpy.context.scene.frame_set(frame=int(frame_ind_blender), subframe=frame_ind_blender - int(frame_ind_blender))

        for joint in joints:
            joint_name = joint.get_name()
            # project to plane at depth = plane_depth along viewing direction
            okp_location = skeleton.poses[frame_ind, :, joint]
            okp_location *= plane_depth / okp_location[2]
            # to blender coordinates
            okp_location = Vector(stealth2blender_coords(okp_location))
            # animate blender object
            okp = bpy.data.objects[joint_name+'.Output.Keypoint']
            okp.location = okp_location
            okp.keyframe_insert(data_path='location')

    # for frame_id, frame_ind in skeleton._frame_ids.items():
    #     frame_time = skeleton.get_time(frame_id) * time_scale
    for frame_ind_blender in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        # frame_ind_blender = frame_time * (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
        bpy.context.scene.frame_set(frame=int(frame_ind_blender), subframe=frame_ind_blender - int(frame_ind_blender))

        for joint in joints:
            joint_name = joint.get_name()

            ikp = bpy.data.objects[joint_name+'.Input.Keypoint']
            okp = bpy.data.objects[joint_name+'.Output.Keypoint']

            offset_cyl = bpy.data.objects[joint_name+'.Offset']
            offset_cyl.location = ((okp.location + ikp.location) / 2) * 1.01
            offset_cyl.scale = (1, 1, (okp.location - ikp.location).magnitude * 1.01)
            new_dir = (okp.location - ikp.location).normalized()
            new_dir.z = -new_dir.z
            old_dir = Vector([0, 0, -1])
            axis = new_dir.cross(old_dir).normalized()
            angle = new_dir.angle(old_dir)
            offset_cyl.rotation_quaternion = Quaternion(axis, angle)

            offset_cyl.keyframe_insert(data_path='location')
            offset_cyl.keyframe_insert(data_path='scale')
            offset_cyl.keyframe_insert(data_path='rotation_quaternion')

    # for joint in joints:
    #     joint_name = joint.get_name()

    #     ikp = bpy.data.objects[joint_name+'.Input.Keypoint']
    #     okp = bpy.data.objects[joint_name+'.Output.Keypoint']

    #     # create connecting line and hook to input/output keypoints
    #     # curve, curvedata = create_polyline(name=joint_name+'.Offset', pts=[tuple(ikp.location), tuple(okp.location)])
    #     curvedata = bpy.data.curves.new(joint_name+'.Offset.Curve', 'CURVE')
    #     curvedata.dimensions = '3D'
    #     spline = curvedata.splines.new('BEZIER')
    #     spline.bezier_points.add(1)
    #     start_point = spline.bezier_points[0]
    #     end_point = spline.bezier_points[0]
    #     start_point.co = ikp.location
    #     start_point.handle_right_type = 'AUTO'
    #     end_point.co = okp.location
    #     end_point.handle_left_type = 'AUTO'


    #     curvedata.bevel_depth = 0.002
    #     curvedata.bevel_resolution = 5
    #     curvedata.fill_mode = 'FULL'

    #     curve = bpy.data.objects.new(joint_name+'.Offset.Object', curvedata)

    #     start_hook = curve.modifiers.new(joint_name+'.Offset.Starthook', 'HOOK')
    #     start_hook.object = ikp
    #     end_hook = curve.modifiers.new(joint_name+'.Offset.Endhook', 'HOOK')
    #     end_hook.object = okp

    #     bpy.context.scene.objects.link(curve)

    #     curve.parent = group
    #     bpy.context.scene.objects.active = curve

    #     bpy.ops.object.mode_set(mode='EDIT')

    #     bpy.ops.object.hook_reset(modifier=joint_name+'.Offset.Starthook')
    #     bpy.ops.object.hook_reset(modifier=joint_name+'.Offset.Endhook')

    #     # start_point = curvedata.splines[0].points[0]
    #     # end_point = curvedata.splines[0].points[1]
    #     start_point = curvedata.splines[0].bezier_points[0]
    #     end_point = curvedata.splines[0].bezier_points[1]

    #     # start_point.select = True
    #     start_point.select_control_point = True
    #     bpy.ops.object.hook_reset(modifier=joint_name+'.Offset.Starthook')
    #     bpy.ops.object.hook_assign(modifier=joint_name+'.Offset.Starthook')
    #     # start_point.select = False
    #     start_point.select_control_point = False

    #     # end_point.select = True
    #     end_point.select_control_point = True
    #     bpy.ops.object.hook_reset(modifier=joint_name+'.Offset.Endhook')
    #     bpy.ops.object.hook_assign(modifier=joint_name+'.Offset.Endhook')
    #     # end_point.select = False
    #     end_point.select_control_point = False

    #     bpy.ops.object.mode_set(mode='OBJECT')    



def import_video(video_filepath):
    # load video
    clip = bpy.data.movieclips.load(video_filepath)

    # show video as background in viewport
    area = next((area for area in bpy.context.screen.areas if area.type == 'VIEW_3D'), None)
    space = next((space for space in area.spaces if space.type == 'VIEW_3D'), None)

    space.show_background_images = True
    bg = space.background_images.new()
    bg.clip = clip
    bg.source = 'MOVIE_CLIP'
    bg.use_camera_clip = False
    bg.opacity = 1.

    # add video as background layer to compositor
    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # for link in [l in links]:
    #     links.remove(link)

    node_render_layers = nodes['Render Layers']

    node_alpha_mul = nodes.new('CompositorNodeMapRange')
    node_alpha_mul.inputs[1].default_value = 0
    node_alpha_mul.inputs[2].default_value = 1
    node_alpha_mul.inputs[3].default_value = 0
    node_alpha_mul.inputs[4].default_value = 1

    node_clip = nodes.new('CompositorNodeMovieClip')
    node_clip.name = 'Video'
    node_clip.clip = clip

    node_alpha = nodes.new('CompositorNodeAlphaOver')

    node_composite = nodes['Composite']

    links.new(node_render_layers.outputs[1], node_alpha_mul.inputs[0]) # scale alpha
    links.new(node_render_layers.outputs[0], node_alpha.inputs[2]) # rendered image
    links.new(node_clip.outputs[0], node_alpha.inputs[1]) # movie clip
    links.new(node_alpha_mul.outputs[0], node_alpha.inputs[0]) # rendered alpha

    links.new(node_alpha.outputs[0], node_composite.inputs[0])

    return bg


def stealth2blender_coords(v):
    return [v[0], v[2], -v[1]]


def parse_args():
    # remove blender arguments
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description='Import Stealth scene into Blender.')
    parser.add_argument('path', type=str, default='./', help='Path to the json scene file.')
    parser.add_argument('--output', type=str, default='', help='Path to the output blender file.')
    parser.add_argument('--video', type=str, default='../input/input_cropped_10fps.mp4', help='Path to the input video, relative to the scene file path.')
    parser.add_argument('--fps', type=int, default=10, help='Fps for the scene, this has to match the video fps if a video is given!')
    parser.add_argument('--intrinsics', type=str, default='../intrinsics.json', help='Path to the json file containing the camera intrinsics, relative to the scene file path.')
    parser.add_argument('--keypoints', type=str, default='', help='Path to the json file containing the 2d keypoints, relative to the scene file path (leave empty for default).')
    parser.add_argument('--localpose', type=str, default='', help='Path to the json file containing the 2d keypoints, relative to the scene file path (leave empty for default).')
    parser.add_argument('--render_width', type=int, default=1920, help='Width of the rendered image (height is determinded from video aspect ratio).')
    # parser.add_argument('--render_height', type=int, default=1080, help='Height of the rendered image.')
    parser.add_argument('--time_scale', type=float, default=1, help='Global scaling of animation time.')
    parser.add_argument('--use_cycles', type=int, default=True, help='Use cycles renderer.')
    parser.add_argument('--quick', action='store_true', help='Quick render')
    parser.add_argument('--candidate', type=str, help='Path to candidate rendering')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    try:
        scene_name = os.path.basename(
            os.path.normpath(os.path.dirname(args.path) + '/..'))
        if scene_name.startswith('opt') or scene_name.startswith('output'):
            scene_name = os.path.basename(
                os.path.normpath(os.path.dirname(args.path) + '/../..'))
        scene_file_name = os.path.basename(args.path)

        video_fpath = os.path.join(os.path.dirname(args.path), args.video)
        intrinsics_fpath = os.path.join(os.path.dirname(args.path), args.intrinsics)
        if not os.path.isfile(intrinsics_fpath):
            intrinsics_fpath = os.path.join(os.path.dirname(args.path),
                                            os.pardir, args.intrinsics)

        if len(args.keypoints) == 0:
            args.keypoints = '../skel_%s_2d_00.json' % scene_name
        keypoint_fpath = os.path.join(os.path.dirname(args.path), args.keypoints)
        if len(args.localpose) == 0:
            args.localpose = '../skel_%s_lfd_orig.json' % scene_name
        localpose_fpath = os.path.join(os.path.dirname(args.path), args.localpose)

        if scene_name in ['lobby12-couch-table', 'lobby18-1', 'lobby19-3'] \
          and not scene_file_name.startswith('skel_tome') \
          and 'Tome3D' not in os.path.split(scene_file_name)[-1] \
          and 'LCRNet3D' not in os.path.split(scene_file_name)[-1]:
            t_scale = 24 / 100  # time given incorrectly as indices into a 4.166666 fps clip?
        else:
            t_scale = 1

        # t_scale = 1/100 # temp

        use_cycles = args.use_cycles

        import_scene(
            scene_filepath=args.path,
            video_filepath=video_fpath,
            keypoint_filepath=keypoint_fpath,
            localpose_filepath=localpose_fpath,
            intrinsics_filepath=intrinsics_fpath,
            # intrinsics_filepath=os.path.join(os.path.dirname(intrinsics_fpath), os.path.basename(intrinsics_fpath)),
            # intrinsics_filepath=None,
            time_scale=t_scale,
            fps=args.fps,
            render_width=args.render_width,
            include_objects=True,
            include_skeleton=True,
            include_video=True,
            include_keypoints=False,
            include_localpose=False,
            sphere_scale=0.5
            )
        
        if True:
            import imapper.blender.replace_objects_w_models
            bpy.ops.file.find_missing_files(directory='/Users/aron/workspace/ucl/stealth/data/models/')
            bpy.ops.file.pack_all()
            
        # Siggraph Asia rendering properties
        if args.quick:
            scene = bpy.context.scene
            # light rays per pixel - increase for less specles
            scene.cycles.samples = 30
            # light ray bounces
            scene.cycles.max_bounces = 4
            # bounces can be set per lamp as well
            bpy.data.lamps['Lamp.001'].cycles.max_bounces = 4
            # resolution
            scene.render.resolution_percentage = 42
            # GPU efficiency (render batch size)
            scene.render.tile_x = 256
            scene.render.tile_y = 256
            # set active camera
            scene.camera = bpy.data.objects['Camera.Viewing']
            
            # probably don't help:
            scene.cycles.caustics_reflective = False
            scene.cycles.caustics_refractive = False

        if len(args.output) > 0:
            # bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=args.output)
            print('Saved to {}'.format(args.output))
            # bpy.ops.wm.quit_blender()
        if hasattr(args, 'candidate') and args.candidate is not None:
            bpy.context.scene.render.filepath = args.candidate
            bpy.context.scene.render.use_compositing = False
            bpy.context.scene.render.resolution_percentage = 100
            bpy.data.objects['floor'].cycles.is_shadow_catcher = True
            bpy.data.objects['Output.Actor.Camera'].rotation_euler[0] = radians(60.)
            bpy.context.scene.camera = bpy.data.objects['Output.Actor.Camera']
            bpy.data.objects['Output.Actor.Camera'].data.ortho_scale = 4.6
            bpy.data.lamps['Lamp.001'].cycles.max_bounces = 10
            scene.cycles.max_bounces = 24
            scene.cycles.samples = 60

    except bdb.BdbQuit as e:
        # exiting from the debugger
        pass
    except SystemExit as e:
        if e.code != 0:
            raise
    except Exception as e:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem()
   
# Render script to call on thorin:
# scene="lobby15"; \
# ROOT="/media/data/amonszpa/stealth/shared/video_recordings/"; \
# /home/amonszpa/blender-2.78c \
#  -b ${ROOT}/${scene}/output/skel_output.blend \
#  -o ${ROOT}/${scene}/output/render/####.png \
#  -E CYCLES \
#  -P /media/data/amonszpa/stealth/data/code/stealth/scripts/thorin/render_blender.py \
#  -- \
#  ${ROOT}/${scene}/output/render/ \
#  3

# scene=lobby15; \
# BLENDER="/Applications/blender/blender.app/Contents/MacOS/blender"; \
# ROOT="/Users/aron/workspace/ucl/stealth/data/video_recordings/"; \
# python3 ${ROOT}/../../scripts/stealth/blender/show_scene.py \
#  --blender ${BLENDER} \
#  --scene ${ROOT}/${scene}/output/skel_output.json \
#  --video ../input/input_cropped.mp4 \
#  --fps 24 \
#  --out ${ROOT}/${scene}/output/skel_output.blend \
#  --quick
