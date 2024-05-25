import random
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation
from utils.math_utils import deg_to_rad


class CooccurrenceRotation(Transformation):
    name = 'co-occurrence_rotation'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        Random value in [120, 140] ∪ [220, 240]
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        original_dim = self.blender_api.dimensions(obj_name)
        z_rotation = random.randint(160, 200)
        x_movement = random.uniform(-1 * original_dim[0] / 2, original_dim[0] / 2)
        y_movement = random.uniform(-1 * original_dim[1] / 2, original_dim[1] / 2)
        result['transform']['location'][0] += x_movement
        result['transform']['location'][1] += y_movement
        result['transform']['location'][2] += 1
        result['transform']['rotation'][2] += deg_to_rad(z_rotation)
        self.execute(result, None)
        intersections = self.blender_api.check_intersections(obj_name)
        if intersections:
            mean_intersection_location = \
                [sum([self.blender_api.location(x)[i] for x in intersections])/len(intersections) for i in range(3)]
            move_vector = self.blender_api.build_movement_vector(mean_intersection_location,
                                                                 self.blender_api.location(obj_name))
            new_location = [self.blender_api.location(obj_name)[i] + (move_vector[i] * 10) for i in range(3)]
            result['transform']['location'][0] = new_location[0]
            result['transform']['location'][1] = new_location[1]
        return result

    def _internal_validate(self, camera_name, metadata, transformation):
        obj_name = transformation['obj_name']
        resolution = self.config['render_resolution']
        is_in_view = obj_name in self.blender_api.models_in_camera_view(camera_name)
        bbox = self.blender_api.bounding_box_2d(camera_name, obj_name, resolution, True)
        valid_bbox = self.blender_api.is_valid_bbox(bbox, resolution, 0.1)
        intersections = self.blender_api.check_intersections(obj_name)
        no_intersection = len(intersections) == 0
        return {
            'in_view': is_in_view,
            'valid_bbox': valid_bbox,
            'no_intersections': no_intersection
        }

