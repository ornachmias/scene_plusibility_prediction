import random
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation


class Intersection(Transformation):
    name = 'intersection'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        Move the object until 30%-50% of the object’s bounding box will overlap with another object.
        Make sure the objects intersect by checking mesh intersections.
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        obj_dim = self.blender_api.dimensions(obj_name)
        x_delta = random.uniform(-1 * obj_dim[0] / 2, obj_dim[0] / 2)
        y_delta = random.uniform(-1 * obj_dim[1] / 2, obj_dim[1] / 2)
        z_delta = random.uniform(-2 * obj_dim[2] / 3, -1 * obj_dim[2] / 3)

        result['transform']['location'][0] += x_delta
        result['transform']['location'][1] += y_delta
        result['transform']['location'][2] += z_delta
        return result

    def _internal_validate(self, camera_name, metadata, transformation):
        obj_name = transformation['obj_name']
        intersections = self.blender_api.check_intersections(obj_name)
        resolution = self.config['render_resolution']
        is_in_view = obj_name in self.blender_api.models_in_camera_view(camera_name)
        bbox = self.blender_api.bounding_box_2d(camera_name, obj_name, resolution, True)
        valid_bbox = self.blender_api.is_valid_bbox(bbox, resolution, 0.01)

        valid_intersections = False
        for intersection in intersections:
            if 'room' in intersection.lower():
                continue

            other_obj_bbox = self.blender_api.bounding_box_2d(camera_name, intersection, resolution, True)
            intersection_percentage = self.calculate_intersection(bbox, other_obj_bbox)
            if intersection_percentage > 0.4:
                valid_intersections = True
                break

        return {
            'in_view': is_in_view,
            'valid_bbox': valid_bbox,
            'valid_intersections': valid_intersections
        }

    @staticmethod
    def calculate_intersection(bbox1, bbox2):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        iob = intersection_area / float(bb1_area)
        return iob

