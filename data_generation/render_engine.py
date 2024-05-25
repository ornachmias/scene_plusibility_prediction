import os.path
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.validation_result import ValidationResult
from utils.file_utils import save_json


class RenderEngine:
    def __init__(self, logger: Logger, config: dict, blender_api: BlenderApi):
        self.logger = logger
        self.config = config
        self.blender_api = blender_api
        self.plausible_dir = os.path.join(self.config['render_dir'], 'plausible')
        os.makedirs(self.plausible_dir, exist_ok=True)
        self.implausible_dir = os.path.join(self.config['render_dir'], 'implausible')
        os.makedirs(self.implausible_dir, exist_ok=True)

    def render(self, scene_metadata, camera_name, output_name, transformations):
        output_dir = self.plausible_dir if transformations is None else self.implausible_dir
        if transformations is not None:
            unique_transformations = list(set([x['transformation_type'] for x in transformations]))
            if len(unique_transformations) == 1:
                output_dir = os.path.join(output_dir, unique_transformations[0])
            else:
                output_dir = os.path.join(output_dir, 'mixed')

        os.makedirs(output_dir, exist_ok=True)
        image_path = self.blender_api.render(output_dir, output_name, camera_name, self.config['render_resolution'])
        metadata_path = os.path.join(output_dir, f'{output_name}.json')
        self.generate_metadata(metadata_path, scene_metadata, image_path, scene_metadata['blend_path'],
                               transformations, camera_name)
        self.logger.info(f'Image rendering completed. image_path={image_path}, metadata_path={metadata_path}')
        return metadata_path, image_path

    def generate_metadata(self, path, scene_metadata, image_path, blend_path, transformations, camera_name):
        visible_objects = self.blender_api.models_in_camera_view(camera_name)
        objects_metadata = [scene_metadata['models'][x] for x in visible_objects]
        metadata = {
            'scene_id': scene_metadata['scene_id'],
            'validation': ValidationResult.not_performed,
            'image_path': image_path,
            'blend_path': blend_path,
            'camera_name': camera_name,
            'visible_objects': visible_objects,
            'objects_metadata': objects_metadata,
            'bbox_data': self.blender_api.extract_bboxes(camera_name, self.config['render_resolution'])
        }

        if transformations is not None:
            metadata['transformations'] = transformations

        save_json(path, metadata)

    @staticmethod
    def render_name(scene_id, camera_name, n_transformed):
        return f'{scene_id}.{camera_name}.{str(n_transformed).zfill(4)}'
