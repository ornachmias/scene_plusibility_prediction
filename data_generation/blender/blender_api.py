import os
import re

import bpy

from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

from data_generation.blender.cameras_api import CamerasApi
from data_generation.blender.objects_api import ObjectsApi


class BlenderApi(ObjectsApi, CamerasApi):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger

    @staticmethod
    def set_resolution(value):
        bpy.context.scene.render.resolution_x = value
        bpy.context.scene.render.resolution_y = value
        bpy.context.scene.render.resolution_percentage = 100

    def use_gpu(self, resolution):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.tile_x = resolution
        bpy.context.scene.render.tile_y = resolution
        self.set_resolution(resolution)
        bpy.data.scenes[0].render.engine = 'CYCLES'
        bpy.data.scenes[0].render.tile_x = resolution
        bpy.data.scenes[0].render.tile_y = resolution

        scene = bpy.context.scene
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences
        prefs.addons['cycles'].preferences.get_devices()
        cprefs = prefs.addons['cycles'].preferences
        # Attempt to set GPU device types if available
        for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
            try:
                cprefs.compute_device_type = compute_device_type
                # self.logger.debug(f'Device found {compute_device_type}')
                break
            except TypeError:
                pass

        for device in cprefs.devices:
            # self.logger.debug(f'Device Name: {device.name}')
            if not re.match('intel', device.name, re.I):
                self.logger.debug(f'Activating {device}')
                device.use = True
            else:
                device.use = False

    def clear_objects(self, exceptions=None):
        self.logger.info(f'Clearing all objects from workspace. Exceptions: {exceptions}')
        if exceptions is None:
            exceptions = []

        bpy.data.scenes.new("Scene")
        bpy.ops.object.select_all(action='DESELECT')

        for o in bpy.data.objects:
            if o.name not in exceptions:
                self.delete(o.name)

    def import_model(self, path, model_id, scale, transform, index=0):
        self.logger.debug(f'Importing model in path: \'{path}\'')
        bpy.ops.import_scene.obj(filepath=path)
        import_name = f'{model_id}_{index}'
        bpy.context.selected_objects[0].name = f'{model_id}_{index}'
        bpy.context.selected_objects[0].data.name = f'{model_id}_{index}'
        bpy.data.objects[import_name]['inst_id'] = index
        bpy.data.objects[import_name].scale = (scale, scale, scale)
        bpy.data.objects[import_name].matrix_world = Matrix(transform)

        bpy.data.objects[import_name].select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        orig_location = list(bpy.data.objects[import_name].location[:])
        orig_rotation = list(bpy.data.objects[import_name].rotation_euler[:])
        orig_scale = list(bpy.data.objects[import_name].scale[:])

        # Due to bug causing the rendered materials to be with 0.0 alpha
        for mat in bpy.data.objects[import_name].data.materials:
            mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = 1

        bpy.data.objects[import_name].select_set(False)
        return import_name, orig_location, orig_rotation, orig_scale

    def import_camera(self, name, eye, target):
        self.logger.debug(f'Importing camera name: \'{name}\'')
        eye = Vector(eye)
        target = Vector(target)

        camera_object_data = bpy.data.cameras.new(name)
        camera_object_data.clip_start = 0.01
        camera_object_data.clip_end = 1000000

        camera_object = bpy.data.objects.new(name, camera_object_data)
        camera_object.location = eye
        self.point_at(camera_object, target)

        bpy.context.scene.collection.objects.link(camera_object)

    def add_light(self, location, energy):
        self.logger.info(f'Adding light object in location={location}')
        light_data = bpy.data.lights.new(name='light', type='POINT')
        light_data.energy = energy
        light_data.shadow_soft_size = 10
        light_object = bpy.data.objects.new(name='light', object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = (location[0], location[1], location[2])
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

    @staticmethod
    def clear_duplicates():
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.editmode_toggle()
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles()
                bpy.ops.object.editmode_toggle()

    @staticmethod
    def save_blend(path):
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=path)
        return path

    def load_blend(self, path):
        if not os.path.isfile(path):
            self.logger.info(f'No blend file was found in \'{path}\'')
            return None

        bpy.ops.wm.open_mainfile(filepath=path)
        bpy.ops.object.mode_set(mode='OBJECT')
        return path

    @staticmethod
    def models():
        models = []
        for ob in bpy.data.objects:
            if ob.type == 'MESH':
                models.append(ob.name)

        return models

    @staticmethod
    def cameras():
        cameras = []
        for ob in bpy.data.objects:
            if ob.type == 'CAMERA':
                cameras.append(ob.name)

        return cameras

    def models_in_camera_view(self, camera_name):
        camera = bpy.data.objects[camera_name]
        bpy.context.scene.camera = camera
        self.update()

        if camera is None:
            raise Exception(f'Camera {camera_name} was not found in scene.')

        models = self.get_objects_in_camera(camera)
        return [x.name for x in models if x.type == 'MESH']

    @staticmethod
    def update():
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

    def set_location(self, name, value):
        super().set_location(name, value)
        self.update()

    def set_hide(self, name, value):
        super().set_hide(name, value)
        self.update()

    def set_rotation(self, name, value):
        super().set_rotation(name, value)
        self.update()

    def set_scale(self, name, value):
        super().set_scale(name, value)
        self.update()

    def adjust_camera(self, name, target):
        cam = bpy.data.objects[name]
        super().point_at(cam, target)
        self.update()

    @staticmethod
    def ray_trace(ray_start, ray_end, target_obj):
        mw = target_obj.matrix_world
        mwi = mw.inverted()

        origin = mwi @ ray_start.matrix_world.translation
        dest = mwi @ ray_end.matrix_world.translation
        direction = (dest - origin).normalized()

        hit, loc, norm, face = target_obj.ray_cast(origin, direction)

        if hit:
            return mw @ loc

        return None

    def is_occluded(self, cam_name, model_name):
        ray_percentage = self.get_raycast_percentage(cam_name, model_name)
        return ray_percentage < 0.005

    def visible_objects(self, cam_name):
        models = self.models()
        visible_models = []
        for model in models:
            is_occluded = self.is_occluded(cam_name, model)
            if not is_occluded:
                visible_models.append(model)

        return visible_models

    @staticmethod
    def check_intersections(name):
        overlapping_obj = []
        original_obj = bpy.data.objects[name]
        original_vert = [original_obj.matrix_world @ v.co for v in original_obj.data.vertices]
        original_poly = [p.vertices for p in original_obj.data.polygons]
        original_bvh = BVHTree.FromPolygons(original_vert, original_poly, epsilon=0.5)

        for current_obj in bpy.data.objects:
            if current_obj.name == name or current_obj.type != 'MESH':
                continue

            current_mat = current_obj.matrix_world
            current_vert = [current_mat @ v.co for v in current_obj.data.vertices]
            current_poly = [p.vertices for p in current_obj.data.polygons]
            current_bvh = BVHTree.FromPolygons(current_vert, current_poly, epsilon=0.5)

            if original_bvh.overlap(current_bvh):
                overlapping_obj.append(current_obj.name)

        return overlapping_obj

    def perform_transformations(self, transformations):
        original_transformations = {}
        if transformations is not None:
            for transformation in transformations:
                name = transformation['model']
                original_transformations[name] = {'location': self.location(name), 'rotation': self.rotation(name)}
                self.set_location(name, transformation['location'])
                self.set_rotation(name, transformation['rotation'])

        return original_transformations

    @staticmethod
    def is_valid_bbox(bbox, resolution, ratio=0.017):
        x_diff = abs(bbox[2] - bbox[0])
        y_diff = abs(bbox[3] - bbox[1])
        return x_diff * y_diff > (resolution * ratio) ** 2

    def extract_bboxes(self, cam_name, resolution):
        bbox_data = {}
        models = self.models()
        for model in models:
            bbox = self.bounding_box_2d(cam_name, model, resolution, True)
            if self.is_valid_bbox(bbox, resolution, 0.):
                bbox_data[model] = bbox

        return bbox_data

    def prepare_render(self, cam_name, resolution):
        self.set_resolution(resolution)
        self.use_gpu(resolution)
        cam = bpy.data.objects[cam_name]
        bpy.context.scene.camera = cam
        bpy.context.view_layer.update()

    def render(self, output_dir, output_name, cam_name, resolution):
        self.prepare_render(cam_name, resolution)
        image_path = os.path.join(output_dir, f'{output_name}.png')
        self.perform_render(image_path)
        return image_path

    def clear_cameras(self, exceptions=None):
        self.logger.info(f'Clearing all cameras from workspace. Exception={exceptions}')
        bpy.ops.object.select_all(action='DESELECT')
        for ob in bpy.data.objects:
            if ob.type == 'CAMERA':
                if exceptions is None or ob.name not in exceptions:
                    bpy.data.objects[ob.name].select_set(True)

        bpy.ops.object.delete()

    @staticmethod
    def build_movement_vector(start, end):
        move_vector = Vector([(x1 - x2) / 2 for x1, x2 in zip(start, end)])
        move_vector.normalize()
        return move_vector

    def point_camera_at(self, camera_name, target_location):
        camera = bpy.context.scene.objects[camera_name]
        super().point_at(camera, target_location)

    def render_model(self, model, output_path, resolution):
        self.clear_cameras()
        self.clear_objects()
        import_name, _, _, _ = self.import_model(model['path'], model['model_id'],
                                                 model['scale'], model['transform'], model['index'])
        self.add_light((0, 0, 150), 50000)
        obj_location = (0, 0, 0)
        self.set_location(import_name, obj_location)
        bb_sides = self.get_3d_bbox_min(import_name) - self.get_3d_bbox_max(import_name)
        (dist_x, dist_y, dist_z) = tuple([abs(c) for c in bb_sides])
        camera_location = (2 * dist_x, 2 * dist_y, 2 * dist_z)
        cam_name = 'cam'
        self.create_camera(camera_location, obj_location, cam_name)
        self.set_resolution(resolution)
        self.use_gpu(resolution)
        cam = bpy.data.objects[cam_name]
        bpy.context.scene.camera = cam
        bpy.context.view_layer.update()
        self.perform_render(output_path)

    @staticmethod
    def perform_render(file_path):
        bpy.context.scene.render.filepath = os.path.abspath(file_path)
        bpy.ops.render.render(write_still=True)

    def get_placed_location(self, obj_name):
        obj = bpy.context.scene.objects[obj_name]
        bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=5, enter_editmode=False, align='WORLD',
                                            location=(obj.location[0], obj.location[1], -50),
                                            scale=(0.1, 0.1, 300))
        cylinder_obj = bpy.context.active_object
        cylinder_obj.name = 'cylinder'
        cylinder_obj.data.name = 'cylinder'
        intersecting_objects = self.check_intersections(cylinder_obj.name)
        intersecting_objects = [x for x in intersecting_objects if x != obj_name and x != cylinder_obj.name]

        cylinder_verts = [cylinder_obj.matrix_world @ v.co for v in cylinder_obj.data.vertices]
        cylinder_poly = [p.vertices for p in cylinder_obj.data.polygons]
        cylinder_bvh = BVHTree.FromPolygons(cylinder_verts, cylinder_poly)

        max_z = 0
        for intersecting_object in intersecting_objects:
            if self.location(intersecting_object)[2] > obj.location[2]:
                continue

            current_obj = bpy.context.scene.objects[intersecting_object]
            current_mat = current_obj.matrix_world
            current_vert = [current_mat @ v.co for v in current_obj.data.vertices]
            current_poly = [p.vertices for p in current_obj.data.polygons]
            current_bvh = BVHTree.FromPolygons(current_vert, current_poly)
            selected_poly = [current_poly[x[0]] for x in current_bvh.overlap(cylinder_bvh)]
            overlap_verts = [current_vert[x[0]] for x in selected_poly]
            available_z_values = [x[2] for x in overlap_verts if x[2] < 100]
            if not available_z_values:
                continue

            current_max_z = max(available_z_values)
            if current_max_z > max_z:
                max_z = current_max_z

        self.delete(cylinder_obj.name)
        if max_z == 0:
            return None

        original_location = self.location(obj_name)
        test_location = [x for x in original_location]
        test_location[2] -= 1
        self.set_location(obj_name, test_location)
        if not self.check_intersections(obj_name):
            self.set_location(obj_name, original_location)
            return None

        return max_z

    def delete_empties(self):
        for x in bpy.data.objects:
            if x.type == 'EMPTY':
                self.delete(x.name)

    def bounding_box_2d(self, camera_name, object_name, resolution, clamp=False):
        self.set_resolution(resolution)
        self.update()
        return super().bounding_box_2d(camera_name, object_name, clamp)

    @staticmethod
    def get_raycast_percentage(camera_name, obj_name, limit=.0001):
        # https://github.com/Danny-Dasilva/Blender-ML
        obj = bpy.data.objects[obj_name]
        cam = bpy.data.objects[camera_name]
        scene = bpy.context.scene

        def bvh_tree_and_vertices_in_world_from_obj(o):
            mWorld = o.matrix_world
            verts_in_world = [mWorld @ vv.co for vv in o.data.vertices]
            tmp_bvh = BVHTree.FromPolygons(verts_in_world, [p.vertices for p in o.data.polygons])
            return tmp_bvh, verts_in_world

        def DeselectEdgesAndPolygons(o):
            for p in o.data.polygons:
                p.select = False
            for e in o.data.edges:
                e.select = False

        # Threshold to test if ray cast corresponds to the original vertex
        viewlayer = bpy.context.view_layer
        # Deselect mesh elements
        DeselectEdgesAndPolygons(obj)
        # In world coordinates, get a bvh tree and vertices
        bvh, vertices = bvh_tree_and_vertices_in_world_from_obj(obj)
        same_count = 0
        count = 0
        for i, v in enumerate(vertices):
            count += 1
            # Get the 2D projection of the vertex
            co2D = world_to_camera_view(scene, cam, v)

            # By default, deselect it
            obj.data.vertices[i].select = False

            # If inside the camera view
            if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0:
                # Try a ray cast, in order to test the vertex visibility from the camera
                location, normal, index, distance, t, ty = scene.ray_cast(viewlayer.depsgraph, cam.location,
                                                                          (v - cam.location).normalized())
                t = (v - normal).length
                if t < limit:
                    same_count += 1
        del bvh
        ray_percent = float(same_count) / float(count)
        return ray_percent

    def remove_scene_dependencies(self):
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                self.unset_parent(obj.name)

    def set_scene_dependencies(self, scene_dependencies):
        for parent_model_name in scene_dependencies:
            children_model_names = scene_dependencies[parent_model_name]
            for child_model_name in children_model_names:
                self.set_parent(child_model_name, parent_model_name)

    def get_gravity_location(self, scene_dependencies, obj_name):
        obj = bpy.data.objects[obj_name]
        try:
            self.remove_scene_dependencies()
            bpy.ops.rigidbody.world_add()
            sc = bpy.context.scene
            sc.rigidbody_world.enabled = True

            collection = bpy.data.collections.get('GravityCollection')
            if collection is None:
                collection = bpy.data.collections.new('GravityCollection')

            sc.rigidbody_world.collection = collection
            sc.rigidbody_world.collection.objects.link(obj)
            for current_obj in bpy.data.objects:
                if current_obj.name not in sc.rigidbody_world.collection.objects:
                    if current_obj.type != 'MESH':
                        continue

                    sc.rigidbody_world.collection.objects.link(current_obj)
                    if current_obj.name != obj_name:
                        current_obj.rigid_body.type = 'PASSIVE'

                current_obj.rigid_body.collision_shape = 'MESH'

            self.update()
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            sc.gravity[2] = -1000

            # Manually baking the scene, to allows early stop and better debugging
            prev_location = None
            for frame in range(1, 250):
                bpy.context.scene.frame_set(frame)
                current_location = obj.matrix_world.to_translation()[:]
                if current_location[2] < 0:
                    return None

                self.logger.debug(f'Gravity location at frame {frame}: {current_location}')
                if prev_location is not None and abs(current_location[2] - prev_location[2]) < 0.5:
                    break

                prev_location = current_location

            new_location = obj.matrix_world.to_translation()[:]
            return new_location
        finally:
            self.undo_gravity(scene_dependencies, obj_name)

    def undo_gravity(self, scene_dependencies, obj_name):
        bpy.ops.ptcache.free_bake_all()

        bpy.ops.object.select_all(action='DESELECT')
        for current_obj in bpy.data.objects:
            if current_obj.type == 'MESH':
                if current_obj.rigid_body is not None:
                    current_obj.select_set(True)
                    bpy.context.view_layer.objects.active = current_obj

        bpy.ops.rigidbody.objects_remove()
        sc = bpy.context.scene
        sc.rigidbody_world.enabled = False

        collection = bpy.data.collections.get('GravityCollection')

        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        bpy.data.collections.remove(collection)

        bpy.ops.rigidbody.world_remove()
        self.set_scene_dependencies(scene_dependencies)
        obj = bpy.data.objects[obj_name]
        obj.animation_data_clear()
        bpy.context.scene.frame_set(1)

    def clamp_to_room(self, obj_name):
        location = self.location(obj_name)

        room_name = None
        for model in self.models():
            if 'room' in model:
                room_name = model
                break

        if room_name is None:
            return location

        room_bbox_min = self.get_3d_bbox_min(room_name)[:]
        room_bbox_max = self.get_3d_bbox_max(room_name)[:]
        new_location = [x for x in location]
        for i in range(3):
            new_location[i] = max(room_bbox_min[i], min(room_bbox_max[i], location[i]))
            obj = bpy.context.scene.objects[obj_name]

            if obj.type == 'MESH':
                if room_bbox_min[i] == new_location[i]:
                    new_location[i] += self.dimensions(obj_name)[i] / 2
                elif room_bbox_max[i] == new_location[i]:
                    new_location[i] -= self.dimensions(obj_name)[i] / 2

        return new_location

