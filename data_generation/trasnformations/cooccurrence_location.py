from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation
from utils.math_utils import random_points_above


class CooccurrenceLocation(Transformation):
    name = 'co-occurrence_location'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        Randomly move an object in the room, then apply gravity to locate it in an implausible way
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        location_found = False
        original_location = self.blender_api.location(obj_name)
        n_iterations = 0
        while not location_found and n_iterations < 10:
            n_iterations += 1
            new_location = random_points_above(original_location, 60, 1)
            if not new_location:
                continue

            self.blender_api.set_location(obj_name, new_location[0])
            new_location = self.blender_api.clamp_to_room(obj_name)
            self.blender_api.set_location(obj_name, new_location)
            new_location = self.blender_api.get_gravity_location(self.dependencies[metadata['scene_id']], obj_name)
            if new_location is None or new_location[2] < 0 or abs(new_location[2] - original_location[2]) < 2.:
                continue

            result['transform']['location'][0] = new_location[0]
            result['transform']['location'][1] = new_location[1]
            result['transform']['location'][2] = new_location[2]
            location_found = True

        if not location_found:
            return None

        return result

    def _internal_validate(self, camera_name, metadata, transformation):
        obj_name = transformation['obj_name']
        resolution = self.config['render_resolution']
        is_in_view = obj_name in self.blender_api.models_in_camera_view(camera_name)
        is_not_occluded = not self.blender_api.is_occluded(camera_name, obj_name)
        bbox = self.blender_api.bounding_box_2d(camera_name, obj_name, resolution, True)
        valid_bbox = self.bbox_visibility(obj_name, camera_name, resolution) > 0.6
        valid_bbox = valid_bbox and self.blender_api.is_valid_bbox(bbox, resolution, 0.02)
        intersections = self.blender_api.check_intersections(obj_name)
        no_intersection = len(intersections) == 0
        return {
            'in_view': is_in_view,
            'not_occluded': is_not_occluded,
            'valid_bbox': valid_bbox,
            'no_intersections': no_intersection
        }

    def bbox_visibility(self, model_name, camera_name, resolution):
        self.blender_api.set_resolution(resolution)
        raw_bbox = list(self.blender_api.bounding_box_2d(camera_name, model_name, resolution))
        raw_area = abs((raw_bbox[3] - raw_bbox[1]) * (raw_bbox[2] - raw_bbox[0]))
        refined_bbox = list(self.blender_api.bounding_box_2d(camera_name, model_name, resolution, True))
        refined_area = abs((refined_bbox[3] - refined_bbox[1]) * (refined_bbox[2] - refined_bbox[0]))
        if raw_area == 0:
            return 0

        return refined_area / raw_area
