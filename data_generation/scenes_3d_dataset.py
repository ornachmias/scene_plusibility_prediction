import json
import os
import shutil
import numpy as np

from utils.file_utils import download, extract, validate_path, load_json


# http://graphics.stanford.edu/projects/actsynth/#data
class Scenes3DDataset:
    models_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/wss.models.zip'
    textures_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/wss.texture.zip'
    categories_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/model_categories.tsv'
    scenes_url = 'http://graphics.stanford.edu/projects/actsynth/datasets/scenes.csv'

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.root_dir = self.config['scenes_3d']['root_dir']
        self.models_dir = os.path.join(self.root_dir, 'models')
        self.scenes_path = os.path.join(self.root_dir, 'scenes.csv')
        self.categories_path = os.path.join(self.root_dir, 'model_categories.tsv')
        resources_dir = self.config['scenes_3d']['resources']
        self.objects_annotations = load_json(os.path.join(resources_dir, 'annotations.json'))
        self.label_to_index = load_json(os.path.join(resources_dir, 'label_to_index.json'))
        self.scenes = {}
        self.categories = {}
        self.scene_ids = []
        self.categories_count = None

    def initialize(self):
        if not self.config['scenes_3d']['skip_download']:
            os.makedirs(self.root_dir, exist_ok=True)
            self.download_and_extract()

        with open(self.scenes_path) as f:
            scenes_lines = f.readlines()

        for line in scenes_lines[1:]:
            separator_index = line.find(',')
            scene_name = line[:separator_index]
            scene_data = line[separator_index + 1:]
            self.scenes[scene_name] = json.loads(scene_data[1:-2])
            self.scene_ids.append(scene_name)

        with open(self.categories_path) as f:
            categories_lines = f.readlines()

        for line in categories_lines:
            split_line = line.split()
            model_id = split_line[0].replace('wss.', '')
            category = split_line[1].lower()
            self.categories[model_id] = category

        self.categories_count = self.get_category_count()

    def set_debug(self):
        self.scenes = {self.config['debug_scene']: self.scenes[self.config['debug_scene']]}

    def download_and_extract(self):
        models_zip_path = os.path.join(self.root_dir, 'wss.models.zip')
        textures_zip_path = os.path.join(self.root_dir, 'wss.texture.zip')
        textures_dir = os.path.join(self.root_dir, 'textures')

        if not os.path.exists(models_zip_path):
            download(self.logger, self.models_url, models_zip_path)

        if not os.path.exists(self.models_dir):
            validate_path(models_zip_path)
            extract(self.logger, models_zip_path)

        if not os.path.exists(textures_zip_path):
            download(self.logger, self.textures_url, textures_zip_path)

        if not os.path.exists(textures_dir):
            validate_path(textures_zip_path)
            extract(self.logger, textures_zip_path)

            # To allow the MTL file automatically load the JPG files we need to copy it to the same directory
            jpg_files = [os.path.join(textures_dir, jpg)
                         for jpg in os.listdir(textures_dir) if jpg.endswith('jpg')]

            self.logger.info(f'Copying {len(jpg_files)} JPG files')
            for jpg_file in jpg_files:
                shutil.copy2(jpg_file, self.models_dir)

        if not os.path.exists(self.categories_path):
            download(self.logger, self.categories_url, self.categories_path)

        if not os.path.exists(self.scenes_path):
            download(self.logger, self.scenes_url, self.scenes_path)

    def get_models(self, scene_id):
        data = self.scenes[scene_id]
        parsed_models = []
        models_data = data['objects']
        if models_data:
            for model_data in models_data:
                model_id = model_data['modelID']
                model_path = os.path.join(self.models_dir, model_id + '.obj')
                if not os.path.exists(model_path):
                    continue

                parsed_models.append({
                    'path': os.path.join(self.models_dir, model_id + '.obj'),
                    'model_id': model_id,
                    'transform': np.reshape(model_data['transform'], (4, 4)).T,
                    'category': self.get_category(model_id)[0],
                    'category_idx': self.get_category(model_id)[1],
                    'index': model_data['index'],
                    'scale': model_data['scale']
                })

        return parsed_models

    def get_cameras(self, scene_id):
        data = self.scenes[scene_id]
        parsed_cameras = []
        cameras_data = data['cameras']
        camera_index = 0
        if cameras_data:
            for camera_data in cameras_data:
                parsed_cameras.append({
                    'index': camera_index,
                    'eye': camera_data['eye'],
                    'target': camera_data['lookAt'],
                    'up': camera_data['up'],
                    'name': 'original_camera_' + str(camera_index).zfill(3)
                })
                camera_index += 1

        return parsed_cameras

    def get_category(self, model_id):
        if 'room' in model_id:
            return 'room', -1

        current_model_id = model_id
        if '_' in current_model_id:
            current_model_id = current_model_id.split('_')[0]

        category = self.objects_annotations[current_model_id]['category']
        if 'sub_category' in self.objects_annotations[current_model_id]:
            category = self.objects_annotations[current_model_id]['sub_category']

        return category, self.label_to_index[category]

    def get_models_by_category(self):
        models = {}
        for scene_id in self.scenes:
            scene_models = self.get_models(scene_id)
            for scene_model in scene_models:
                if scene_model['category'] not in models:
                    models[scene_model['category']] = []

                if scene_model['path'] not in [x['path'] for x in models[scene_model['category']]]:
                    models[scene_model['category']].append(scene_model)

        return models

    def get_category_count(self):
        categories_count = {}
        for scene_id in self.scenes:
            for model in self.get_models(scene_id):
                category = model['category']
                if category not in categories_count:
                    categories_count[category] = 0

                categories_count[category] += 1

        return categories_count



