import bpy
from mathutils import Vector, Matrix

from logging import Logger


class ObjectsApi:
    def __init__(self, logger: Logger):
        self.logger = logger

    @staticmethod
    def get_lowest_point(obj_name):
        obj = bpy.data.objects[obj_name]
        lowest_pt = min([(obj.matrix_world @ v.co).z for v in obj.data.vertices])
        return lowest_pt

    @staticmethod
    def dimensions(name):
        return list(bpy.data.objects[name].dimensions[:])

    def get_3d_bbox_min(self, name):
        return self.get_min(self.bounding_box_3d(name))

    def get_3d_bbox_max(self, name):
        return self.get_max(self.bounding_box_3d(name))

    @staticmethod
    def get_min(bound_box):
        min_x = min([bound_box[i][0] for i in range(0, 8)])
        min_y = min([bound_box[i][1] for i in range(0, 8)])
        min_z = min([bound_box[i][2] for i in range(0, 8)])
        return Vector((min_x, min_y, min_z))

    @staticmethod
    def get_max(bound_box):
        max_x = max([bound_box[i][0] for i in range(0, 8)])
        max_y = max([bound_box[i][1] for i in range(0, 8)])
        max_z = max([bound_box[i][2] for i in range(0, 8)])
        return Vector((max_x, max_y, max_z))

    @staticmethod
    def delete(name):
        object_to_delete = bpy.data.objects[name]
        bpy.data.objects.remove(object_to_delete, do_unlink=True)

    @staticmethod
    def set_parent(child_name, parent_name):
        parent = bpy.data.objects[parent_name]
        child = bpy.data.objects[child_name]
        child.parent = parent
        child.matrix_parent_inverse = parent.matrix_world.inverted()

    @staticmethod
    def unset_parent(child_name):
        child = bpy.data.objects[child_name]
        parented_wm = child.matrix_world.copy()
        parent_name = child.parent
        if parent_name is not None:
            parent_name = parent_name.name

        child.parent = None
        child.matrix_world = parented_wm
        return parent_name

    @staticmethod
    def get_children(obj_name):
        children = []
        for obj in bpy.data.objects:
            if obj.parent is not None and obj.parent.name == obj_name:
                children.append(obj.name)

        return children

    @staticmethod
    def location(name):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        mat = bpy.data.objects[name].matrix_world
        return list(mat.to_translation()[:])

    @staticmethod
    def set_location(name, value):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.data.objects[name].location = value

    @staticmethod
    def set_hide(name, value):
        bpy.data.objects[name].hide_viewport = value
        bpy.data.objects[name].hide_render = value

    @staticmethod
    def set_rotation(name, value):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.data.objects[name].rotation_euler = value

    @staticmethod
    def rotation(name):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        return list(bpy.data.objects[name].rotation_euler[:])

    @staticmethod
    def set_scale(name, value):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.data.objects[name].scale = value

    @staticmethod
    def scale(name):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        return list(bpy.data.objects[name].scale[:])

    @staticmethod
    def fix_normals(model_name):
        obj = bpy.data.objects[model_name]
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()

    @staticmethod
    def create_empty_sphere(name, location):
        bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=location, scale=(1, 1, 1))
        obj_1_rep = bpy.context.active_object
        obj_1_rep.name = name

    @staticmethod
    def duplicate(obj_name, index=0):
        copy_name = f'duplicate_{index}_{obj_name}'
        obj = bpy.data.objects[obj_name]
        obj_copy = obj.copy()
        obj_copy.data = obj_copy.data.copy()
        obj_copy.name = copy_name
        obj_copy.data.name = copy_name
        bpy.context.scene.collection.objects.link(obj_copy)
        return obj_copy.name

    @staticmethod
    def bounding_box_3d(object_name):
        ob = bpy.data.objects[object_name]
        return [ob.matrix_world @ Vector(corner) for corner in ob.bound_box]

    def bounding_box_2d(self, camera_name, object_name, resolution, clamp=False):
        cam_ob = bpy.data.objects[camera_name]
        me_ob = bpy.data.objects[object_name]
        scene = bpy.context.scene
        mat = cam_ob.matrix_world.normalized().inverted()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_eval = me_ob.evaluated_get(depsgraph)
        me = mesh_eval.to_mesh()
        me.transform(me_ob.matrix_world)
        me.transform(mat)

        camera = cam_ob.data
        frame = [-v for v in camera.view_frame(scene=scene)[:3]]
        camera_persp = camera.type != 'ORTHO'

        lx = []
        ly = []

        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if camera_persp:
                if z == 0.0:
                    lx.append(0.5)
                    ly.append(0.5)
                elif z <= 0.0:
                    continue
                else:
                    frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        if not lx:
            lx.append(0)
        if not ly:
            ly.append(0)

        if clamp:
            min_x = self.clamp(min(lx), 0.0, 1.0)
            max_x = self.clamp(max(lx), 0.0, 1.0)
            min_y = self.clamp(min(ly), 0.0, 1.0)
            max_y = self.clamp(max(ly), 0.0, 1.0)
        else:
            min_x = min(lx)
            max_x = max(lx)
            min_y = min(ly)
            max_y = max(ly)

        mesh_eval.to_mesh_clear()

        r = scene.render
        fac = r.resolution_percentage * 0.01
        dim_x = r.resolution_x * fac
        dim_y = r.resolution_y * fac

        # Sanity check
        if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
            return 0, 0, 0, 0

        return (
            round(min_x * dim_x),  # X
            round(dim_y - max_y * dim_y),  # Y
            round(max_x * dim_x),  # Width
            round(dim_y - min_y * dim_y)  # Height
        )

    @staticmethod
    def clamp(x, minimum, maximum):
        return max(minimum, min(x, maximum))