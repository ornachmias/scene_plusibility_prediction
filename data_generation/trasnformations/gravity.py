import math
import random
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation


class Gravity(Transformation):
    name = 'gravity'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        Get the size of the largest dimension of the object, mark it as X.
        Get a random value in range [X, 2X] and move the object in the air by this value.
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        max_box_size = math.ceil(max(self.blender_api.dimensions(obj_name)))
        z_diff = random.uniform(max_box_size, 2 * max_box_size)
        result['transform']['location'][2] += z_diff
        return result

    def _internal_validate(self, camera_name, metadata, transformation):
        obj_name = transformation['obj_name']
        resolution = self.config['render_resolution']
        is_in_view = obj_name in self.blender_api.models_in_camera_view(camera_name)
        is_not_occluded = not self.blender_api.is_occluded(camera_name, obj_name)
        bbox = self.blender_api.bounding_box_2d(camera_name, obj_name, resolution, True)
        valid_bbox = self.blender_api.is_valid_bbox(bbox, resolution)
        intersections = self.blender_api.check_intersections(obj_name)
        no_intersection = len(intersections) == 0
        return {
            'in_view': is_in_view,
            'not_occluded': is_not_occluded,
            'valid_bbox': valid_bbox,
            'no_intersections': no_intersection
        }


