import random
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.transformation import Transformation


class Size(Transformation):
    name = 'size'

    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        super().__init__(self.name, logger, config, blender_api, scenes_3d)

    def find(self, obj_name, metadata):
        """
        On object of size X, set the new dimension to random value in range [3X, 4X]
        """
        super().find(obj_name, metadata)
        result = self._build_transformation_result(obj_name)
        original_lowest_point = self.blender_api.get_lowest_point(obj_name)
        scale = random.uniform(*random.choice([(0.3, 0.5), (2, 3)]))

        for i, x in enumerate(result['transform']['scale']):
            result['transform']['scale'][i] = scale * x

        self.execute(result, None)
        current_lowest_point = self.blender_api.get_lowest_point(obj_name)
        z_location = result['original']['location'][2] + (original_lowest_point - current_lowest_point)
        result['transform']['location'] = [result['original']['location'][0],
                                           result['original']['location'][1], z_location]
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

    def execute(self, transformation, previous_transformations):
        obj_name = transformation['obj_name']
        self._perform_transformation(obj_name, transformation['transform'])
        scale_factor = transformation['transform']['scale'][0] / transformation['original']['scale'][0]
        children = self.blender_api.get_children(obj_name)

        for child in children:
            obj_scale = self.blender_api.scale(child)
            self.blender_api.set_scale(child, [x / scale_factor for x in obj_scale])

        if previous_transformations is not None:
            for previous_transformation in previous_transformations:
                self._perform_transformation(previous_transformation['obj_name'], previous_transformation['transform'])

    def revert(self, transformation):
        obj_name = transformation['obj_name']
        scale_factor = transformation['transform']['scale'][0] / transformation['original']['scale'][0]
        children = self.blender_api.get_children(obj_name)
        for child in children:
            obj_scale = self.blender_api.scale(child)
            self.blender_api.set_scale(child, [x * scale_factor for x in obj_scale])

        self._perform_transformation(obj_name, transformation['original'])



