import os
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from utils.file_utils import save_json, load_json


class Builder:
    def __init__(self, logger: Logger, config, scenes_3d: Scenes3DDataset):
        self.logger = logger
        self.config = config
        self.scenes_3d = scenes_3d
        self.blender_api = BlenderApi(logger)

        resources_dir = self.config['scenes_3d']['resources']
        self.dependencies = load_json(os.path.join(resources_dir, 'dependencies.json'))
        self.repositions = load_json(os.path.join(resources_dir, 'reposition.json'))
        self.blend_dir = self.config['blend_dir']
        self.metadata_dir = self.config['metadata_dir']
        os.makedirs(self.blend_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def import_models(self, scene_id):
        models = self.scenes_3d.get_models(scene_id)
        imported_models = {}
        for model in models:
            if not os.path.isfile(model['path']):
                self.logger.warning(f'Could not find path {model["path"]} for model {model["model_id"]}')
                continue

            import_name, orig_location, orig_rotation, orig_scale = \
                self.blender_api.import_model(model['path'], model['model_id'],
                                              model['scale'], model['transform'], model['index'])
            imported_models[import_name] = {
                'index': model['index'],
                'location': orig_location,
                'rotation': orig_rotation,
                'scale': orig_scale,
                'category': model['category'],
                'category_idx': model['category_idx'],
                'name': import_name,
                'id': model['model_id']
            }

        return imported_models

    def import_cameras(self, scene_id):
        base_cameras = self.scenes_3d.get_cameras(scene_id)
        for base_camera in base_cameras:
            self.blender_api.import_camera(base_camera['name'], base_camera['eye'], base_camera['target'])

        cameras = [{'name': x['name'], 'location': x['eye'], 'target': x['target']} for x in base_cameras]
        return cameras

    def build_scene(self, scene_id):
        self.blender_api.clear_objects()
        models = self.import_models(scene_id)
        cameras = self.import_cameras(scene_id)

        room_center = [models[x]['location'] for x in models if 'room' in x][0]
        light_location = [room_center[0], room_center[1], room_center[2] * 2]
        self.blender_api.add_light(light_location, self.config['scene_build']['light_energy'])
        self.blender_api.clear_duplicates()
        return models, cameras, room_center

    def build_scenes(self):
        for scene_id in self.scenes_3d.scenes:
            blend_output_path = os.path.join(self.blend_dir, f'{scene_id}.blend')
            metadata_output_path = os.path.join(self.metadata_dir, f'{scene_id}.json')

            if os.path.isfile(blend_output_path):
                self.logger.info(f'{blend_output_path} already exists, skipping.')
                continue

            try:
                models, cameras, room_center = self.build_scene(scene_id)
                metadata = {
                    'scene_id': scene_id,
                    'blend_path': blend_output_path,
                    'room_center': room_center,
                    'models': models,
                    'cameras': {
                        'original': cameras,
                        'generated': []
                    }
                }

                save_json(metadata_output_path, metadata)
                self.post_process(scene_id)
                self.blender_api.save_blend(blend_output_path)
                self.logger.info(f'Blend for scene {scene_id} was saved to {blend_output_path}')
                self.logger.info(f'Metadata for scene {scene_id} was saved to {metadata_output_path}')
            except Exception as e:
                self.logger.exception(f'Failed to build scene {scene_id}. Reason: {e}')

    def post_process(self, scene_id):
        self.reposition_models(scene_id)
        self.set_dependencies(scene_id)
        self.fix_normals()

    def reposition_models(self, scene_id):
        if scene_id not in self.repositions:
            return

        for model_name in self.repositions[scene_id]:
            if 'location' in self.repositions[scene_id][model_name]:
                self.blender_api.set_location(model_name, self.repositions[scene_id][model_name]['location'])

            if 'rotation' in self.repositions[scene_id][model_name]:
                self.blender_api.set_rotation(model_name, self.repositions[scene_id][model_name]['rotation'])

            if 'scale' in self.repositions[scene_id][model_name]:
                self.blender_api.set_scale(model_name, self.repositions[scene_id][model_name]['scale'])

    def set_dependencies(self, scene_id):
        scene_dependencies = self.dependencies[scene_id]
        for parent_model_name in scene_dependencies:
            children_model_names = scene_dependencies[parent_model_name]
            for child_model_name in children_model_names:
                self.blender_api.set_parent(child_model_name, parent_model_name)

    def fix_normals(self):
        models = self.blender_api.models()
        for model_name in models:
            self.blender_api.fix_normals(model_name)





