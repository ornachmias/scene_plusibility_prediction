import bpy
from mathutils import Vector, Matrix
from mathutils.geometry import normal

from logging import Logger


class CamerasApi:
    def __init__(self, logger: Logger):
        self.logger = logger

    @staticmethod
    def point_at(obj, target, roll=0):
        """
        Rotate obj to look at target

        :arg obj: the object to be rotated. Usually the camera
        :arg target: the location (3-tuple or Vector) to be looked at
        :arg roll: The angle of rotation about the axis from obj to target in radians.

        Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)
        """
        if not isinstance(target, Vector):
            target = Vector(target)
        loc = obj.location
        # direction points from the object to the target
        direction = target - loc
        quat = direction.to_track_quat('-Z', 'Y')
        quat = quat.to_matrix().to_4x4()
        roll_matrix = Matrix.Rotation(roll, 4, 'Z')

        # remember the current location, since assigning to obj.matrix_world changes it
        loc = loc.to_tuple()
        obj.matrix_world = quat @ roll_matrix
        obj.location = loc
        bpy.context.view_layer.update()

    def get_objects_in_camera(self, camera):
        scene = bpy.context.scene
        origin = camera.matrix_world.to_translation()
        planes = self.camera_as_planes(scene, camera)
        objects_in_view = self.objects_in_planes(scene.objects, planes, origin)
        return objects_in_view

    @staticmethod
    def camera_as_planes(scene, obj):
        """
        Return planes in world-space which represent the camera view bounds.
        """

        camera = obj.data
        # normalize to ignore camera scale
        matrix = obj.matrix_world
        frame = [matrix @ v for v in camera.view_frame(scene=scene)]
        origin = matrix.to_translation()

        planes = []
        is_persp = (camera.type != 'ORTHO')
        for i in range(4):
            # find the 3rd point to define the planes direction
            if is_persp:
                frame_other = origin
            else:
                frame_other = frame[i] + matrix.col[2].xyz

            n = normal((frame_other, frame[i - 1], frame[i]))
            d = -n.dot(frame_other)
            planes.append((n, d))

        if not is_persp:
            # add a 5th plane to ignore objects behind the view
            n = normal((frame[0], frame[1], frame[2]))
            d = -n.dot(origin)
            planes.append((n, d))

        return planes

    @staticmethod
    def side_of_plane(p, v):
        return p[0].dot(v) + p[1]

    def is_segment_in_planes(self, p1, p2, planes):
        dp = p2 - p1

        p1_fac = 0.0
        p2_fac = 1.0

        for p in planes:
            div = dp.dot(p[0])
            if div != 0.0:
                t = -self.side_of_plane(p, p1)
                if div > 0.0:
                    # clip p1 lower bounds
                    if t >= div:
                        return False
                    if t > 0.0:
                        fac = (t / div)
                        p1_fac = max(fac, p1_fac)
                        if p1_fac > p2_fac:
                            return False
                elif div < 0.0:
                    # clip p2 upper bounds
                    if t > 0.0:
                        return False
                    if t > div:
                        fac = (t / div)
                        p2_fac = min(fac, p2_fac)
                        if p1_fac > p2_fac:
                            return False
        p1_clip = p1.lerp(p2, p1_fac)
        p2_clip = p1.lerp(p2, p2_fac)
        epsilon = -0.5
        return all(self.side_of_plane(p, p1_clip) > epsilon and
                   self.side_of_plane(p, p2_clip) > epsilon for p in planes)

    @staticmethod
    def point_in_object(obj, pt):
        xs = [v[0] for v in obj.bound_box]
        ys = [v[1] for v in obj.bound_box]
        zs = [v[2] for v in obj.bound_box]
        pt = obj.matrix_world.inverted() @ pt
        return (min(xs) <= pt.x <= max(xs) and
                min(ys) <= pt.y <= max(ys) and
                min(zs) <= pt.z <= max(zs))

    def object_in_planes(self, obj, planes):
        matrix = obj.matrix_world
        box = [matrix @ Vector(v) for v in obj.bound_box]
        epsilon = -0.00001
        for v in box:
            if all(self.side_of_plane(p, v) > epsilon for p in planes):
                # one point was in all planes
                return True

        # possible one of our edges intersects
        edges = ((0, 1), (0, 3), (0, 4), (1, 2),
                 (1, 5), (2, 3), (2, 6), (3, 7),
                 (4, 5), (4, 7), (5, 6), (6, 7))
        if any(self.is_segment_in_planes(box[e[0]], box[e[1]], planes)
               for e in edges):
            return True

        return False

    def objects_in_planes(self, objects, planes, origin):
        """
        Return all objects which are inside (even partially) all planes.
        """
        return [obj for obj in objects
                if self.point_in_object(obj, origin) or self.object_in_planes(obj, planes)]

    def create_camera(self, location, target, name):
        camera_location = Vector(location)
        camera_target = Vector(target)

        camera_data = bpy.data.cameras.new(name)
        camera_data.clip_start = 0.01
        camera_data.clip_end = 1000000
        camera_data.lens = 30

        camera_object = bpy.data.objects.new(name, camera_data)
        camera_object.location = camera_location
        self.point_at(camera_object, camera_target)

        bpy.context.scene.collection.objects.link(camera_object)
        return camera_object
