import random
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation
from utils.math_utils import deg_to_rad


class Pose(Transformation):
    name = 'pose'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        Randomly move an object in the room, then apply gravity to locate it in an implausible way
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        original_location = self.blender_api.location(obj_name)
        original_lowest_point = self.blender_api.get_lowest_point(obj_name)

        rotation_values = [random.randint(0, 360) for _ in range(3)]
        for i in range(3):
            result['transform']['rotation'][i] += deg_to_rad(rotation_values[i])

        self.blender_api.set_rotation(obj_name, rotation_values)
        current_lowest_point = self.blender_api.get_lowest_point(obj_name)
        z_location = result['original']['location'][2] + (original_lowest_point - current_lowest_point)
        original_location[2] = z_location
        self.blender_api.set_location(obj_name, original_location)
        result['transform']['location'][2] = z_location
        return result

    def _internal_validate(self, camera_name, metadata, transformation):
        obj_name = transformation['obj_name']
        resolution = self.config['render_resolution']
        is_in_view = obj_name in self.blender_api.models_in_camera_view(camera_name)
        is_not_occluded = self.blender_api.get_raycast_percentage(camera_name, obj_name) > 0.3
        bbox = self.blender_api.bounding_box_2d(camera_name, obj_name, resolution, True)
        valid_bbox = self.blender_api.is_valid_bbox(bbox, resolution, 0.01)
        intersections = self.blender_api.check_intersections(obj_name)
        no_intersection = len(intersections) == 0
        return {
            'in_view': is_in_view,
            'not_occluded': is_not_occluded,
            'valid_bbox': valid_bbox,
            'no_intersections': no_intersection
        }

