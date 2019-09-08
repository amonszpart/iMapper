from math import radians

import bpy
import os
from mathutils import Vector, Matrix


def get_max(ob):
    mx = Vector((-1000., -1000., -1000.))
    for vx in ob.data.vertices:
        p = ob.matrix_world * vx.co
        mx.x = max(mx.x, p.x)
        mx.y = max(mx.y, p.y)
        mx.z = max(mx.z, p.z)
    
    return mx


floor = bpy.data.objects['floor']
bpy.ops.object.select_all(action="DESELECT")
for ob in bpy.data.objects:
    if not ob.name.startswith('Object.'):
        continue
    parts = ob.name[7:].split('_')
    oid = int(parts[0])
    typ = parts[1]
    part = parts[2]
    if '-' in part:
        part = part[:part.index('-')]
    name_model = None
    if typ == 'table':
        if part != 'top':
            continue
        else:
            name_model = 'table'
    elif typ == 'couch' or typ == 'chair':
        if part != 'seat':
            continue
        else:
            name_model = typ
    elif typ == 'shelf':
        name_model = 'shelf'
    assert name_model is not None, "Nooo: {}".format(parts)
    
    name_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, os.pardir, 'models',
                              '{}.obj'.format(name_model))
    bpy.ops.import_scene.obj(filepath=name_model, use_split_objects=True)
    model = bpy.context.selected_objects[-1]
    model.name = 'Model.{}_{}'.format(parts[0], typ)
    print(name_model, model)
    model.location = ob.location
    model.rotation_euler = ob.rotation_euler
    bpy.context.scene.update()
    if typ == 'shelf':
        model.matrix_world *= Matrix.Rotation(radians(90), 4, 'X')
    elif typ == 'couch':
        # model.rotation_euler.z = 3.14159 + ob.rotation_euler.z
        model.matrix_world *= Matrix.Rotation(radians(-90), 4, 'X')
        bpy.context.scene.update()
        model.dimensions.z = ob.dimensions.y + 0.2  # slide bwards
        bpy.context.scene.update()
        model.dimensions.x = ob.dimensions.x + 0.26
        bpy.context.scene.update()
        mx = get_max(ob)
        model.dimensions.y = 0.4925 / (mx.z - floor.location.z)
        bpy.ops.object.select_pattern(pattern='floor', extend=True)
        bpy.context.scene.objects.active = floor
        bpy.ops.object.align(align_mode='OPT_3', align_axis={'Z'})
        model.location.z += floor.dimensions.z
        bpy.context.scene.update()
        # compensate for backwards slide:
        model.location -= model.matrix_world.col[2].xyz * 0.125
    elif typ == 'chair':
        # model.rotation_euler.z = ob.rotation_euler.z
        model.matrix_world *= Matrix.Rotation(radians(-90), 4, 'X')
        bpy.context.scene.update()
        model.dimensions.z = ob.dimensions.y
        bpy.context.scene.update()
        model.dimensions.x = ob.dimensions.x
        bpy.context.scene.update()
        mx = get_max(ob)
        model.dimensions.y = 0.24 / (mx.z - floor.location.z)
        bpy.ops.object.select_pattern(pattern='floor', extend=True)
        bpy.context.scene.objects.active = floor
        bpy.ops.object.align(align_mode='OPT_3', align_axis={'Z'})
        model.location.z += floor.dimensions.z
    elif typ == 'table':
        # model.rotation_euler.z = ob.rotation_euler.z
        model.matrix_world *= Matrix.Rotation(radians(-90), 4, 'X')
        bpy.context.scene.update()
        model.dimensions.z = ob.dimensions.y
        bpy.context.scene.update()
        model.dimensions.x = ob.dimensions.x
        bpy.context.scene.update()
        mx = get_max(ob)
        model.dimensions.y = 0.5 / (mx.z - floor.location.z)
        bpy.ops.object.select_pattern(pattern='floor', extend=True)
        bpy.context.scene.objects.active = floor
        bpy.ops.object.align(align_mode='OPT_3', align_axis={'Z'})
        model.location.z += floor.dimensions.z
