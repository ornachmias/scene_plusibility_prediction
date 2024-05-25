import os
from abc import ABC, abstractmethod
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from utils.file_utils import load_json


class Transformation(ABC):
    def __init__(self, name, logger: Logger, config: dict, blender_api: BlenderApi, scenes_3d: Scenes3DDataset):
        self.name = name
        self.logger = logger
        self.config = config
        self.blender_api = blender_api
        self.scenes_3d = scenes_3d
        resources_dir = self.config['scenes_3d']['resources']
        self.dependencies = load_json(os.path.join(resources_dir, 'dependencies.json'))

    @abstractmethod
    def find(self, obj_name, metadata):
        self.logger.debug(f'Finding transformation={self.name} for obj_name={obj_name}')

    @abstractmethod
    def _internal_validate(self, camera_name, metadata, transformation):
        raise NotImplemented("internal_validate should be implemented!")

    def validate(self, camera_name, metadata, transformation, previous_transformations):
        if transformation is None or not transformation['transform']:
            return False

        obj_name = transformation['obj_name']
        self.execute(transformation, previous_transformations)
        self.blender_api.prepare_render(camera_name, self.config['render_resolution'])
        validation_dict = self._internal_validate(camera_name, metadata, transformation)
        log_str = f'obj_name: {obj_name}, category: {self.scenes_3d.get_category(obj_name)[0]}, type:{self.name}, '
        is_valid = True
        log_str += ', '.join([f'{k}: {validation_dict[k]}' for k in validation_dict])

        for k in validation_dict:
            is_valid = is_valid and validation_dict[k]

        self.logger.info(log_str)
        return is_valid

    def execute(self, transformation, previous_transformations):
        obj_name = transformation['obj_name']
        self._perform_transformation(obj_name, transformation['transform'])
        if previous_transformations is not None:
            for previous_transformation in previous_transformations:
                self._perform_transformation(previous_transformation['obj_name'], previous_transformation['transform'])

    def revert(self, transformation):
        obj_name = transformation['obj_name']
        self._perform_transformation(obj_name, transformation['original'])

    def _build_transformation_result(self, obj_name):
        return {
            'obj_name': obj_name,
            'transformation_type': self.name,
            'original': {
                'location': self.blender_api.location(obj_name),
                'rotation': self.blender_api.rotation(obj_name),
                'scale': self.blender_api.scale(obj_name)
            },
            'transform': {
                'location': self.blender_api.location(obj_name),
                'rotation': self.blender_api.rotation(obj_name),
                'scale': self.blender_api.scale(obj_name)
            }
        }

    def _perform_transformation(self, obj_name, transformation):
        self.blender_api.set_location(obj_name, transformation['location'])
        self.blender_api.set_rotation(obj_name, transformation['rotation'])
        self.blender_api.set_scale(obj_name, transformation['scale'])

