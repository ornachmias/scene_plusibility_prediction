import os.path
from pathlib import Path

from utils import file_utils


# Based on dataset created by http://graphics.stanford.edu/~mdfisher/sceneSynthesis.html
class SceneSynthesisDataset:
    dataset_url = 'http://graphics.stanford.edu/~mdfisher/papers/sceneSynthesisDatabase.zip'

    def __init__(self, logger, data_dir):
        self.root_dir = data_dir
        self.logger = logger
        self.data_dir = os.path.join(self.root_dir, 'scene_synthesis')
        self.raw_data_dir = os.path.join(self.data_dir, 'databaseFull')
        self.models_dir = os.path.join(self.raw_data_dir, 'models')
        self.scenes_dir = os.path.join(self.raw_data_dir, 'scenes')
        self.tags = None
        self.names = None
        self.scenes_dir = None
        self.models_dir = None

    def initialize(self, download_content):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, 'sceneSynthesisDatabase.zip')

        if not download_content:
            return

        if not os.path.isfile(file_path):
            file_utils.download(self.logger, self.dataset_url, file_path)

        if not os.path.isdir(self.raw_data_dir):
            file_utils.extract(self.logger, file_path)

        self.names = self._parse_names_file(os.path.join(self.raw_data_dir, 'fields', 'names.txt'))
        self.tags = self._parse_tags_file(os.path.join(self.raw_data_dir, 'fields', 'tags.txt'))

    def get_scene(self, scene_id):
        path = self._get_scene_path(scene_id)
        data = self._parse_scene_file(path)
        return data

    def _parse_scene_file(self, file_path):
        scene_id = Path(file_path).stem

        with open(file_path) as f:
            lines = f.readlines()

        models = self._parse_models(lines)
        return {'scene_id': scene_id, 'models': models}

    def get_scenes(self):
        scenes = []
        for file_name in os.listdir(self.scenes_dir):
            path = os.path.join(self.scenes_dir, file_name)
            if os.path.isfile(path) and path.endswith('.txt'):
                scenes.append(file_name.removesuffix('.txt'))

        return scenes

    def _get_scene_path(self, scene_id):
        return os.path.join(self.scenes_dir, scene_id + '.txt')

    @staticmethod
    def _parse_tags_file(file_path):
        model_tags = {}
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            line_split = [x.strip() for x in line.split('|')]
            if len(line_split) < 2:
                model_tags[line] = None
            elif len(line_split) == 2 and line_split[1] == '*':
                model_tags[line_split[0]] = None
            else:
                model_id = line_split[0]
                tags = [x.strip() for x in line_split[1:]]
                model_tags[model_id] = tags

        return model_tags

    @staticmethod
    def _parse_names_file(file_path):
        model_names = {}
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            line_split = [x.strip() for x in line.split('|')]
            if len(line_split) < 2:
                model_names[line] = None
            else:
                model_id = line_split[0]
                names = [x.strip() for x in line_split[1].split()]
                model_names[model_id] = names

        return model_names

    def _parse_models(self, lines):
        parsing_dict = self._building_parsing_dict()

        models = []
        current_model = None
        for line in lines:
            if self._is_new_model(line):
                if current_model is not None:
                    models.append(current_model)

                current_model = {}

            for parsing_key in parsing_dict:
                if line.startswith(parsing_key):
                    current_model.update(parsing_dict[parsing_key](line))
                    break

        return models

    def _building_parsing_dict(self):
        parsing_dict = {
            'newModel': self._new_model,
            'parentIndex': self._parent_index,
            'children': self._children,
            'parentMaterialGroupIndex': self._parent_material_group,
            'parentTriangleIndex': self._parent_triangle,
            'parentUV': self._parent_uv,
            'parentContactPosition': self._parent_contact_position,
            'parentContactNormal': self._parent_contact_normal,
            'parentOffset': self._parent_offset,
            'scale': self._scale,
            'transform': self._transform
        }

        return parsing_dict

    @staticmethod
    def _is_new_model(line: str):
        return line.startswith('newModel')

    def _new_model(self, line: str):
        line_split = line.split()
        index = line_split[1].strip()
        model_id = line_split[2].strip()
        tags = self.tags.get(model_id, None)
        names = self.names.get(model_id, None)
        return {'index': int(index), 'model_id': model_id, 'tags': tags, 'names': names}

    @staticmethod
    def _parent_index(line: str):
        line_split = line.split()
        parent_index = line_split[1].strip()
        return {'parent_index': int(parent_index)}

    @staticmethod
    def _children(line: str):
        line_split = line.split()
        return {'children': [int(x.strip()) for x in line_split[1:]]}

    @staticmethod
    def _parent_material_group(line: str):
        line_split = line.split()
        parent_material_group = line_split[1].strip()
        return {'parent_material_group': int(parent_material_group)}

    @staticmethod
    def _parent_triangle(line: str):
        line_split = line.split()
        parent_triangle = line_split[1].strip()
        return {'parent_triangle': int(parent_triangle)}

    @staticmethod
    def _parent_uv(line: str):
        line_split = line.split()
        return {'parent_uv': [float(x.strip()) for x in line_split[1:]]}

    @staticmethod
    def _parent_contact_position(line: str):
        line_split = line.split()
        return {'parent_contact_position': [float(x.strip()) for x in line_split[1:]]}

    @staticmethod
    def _parent_contact_normal(line: str):
        line_split = line.split()
        return {'parent_contact_normal': [float(x.strip()) for x in line_split[1:]]}

    @staticmethod
    def _parent_offset(line: str):
        line_split = line.split()
        return {'parent_offset': [float(x.strip()) for x in line_split[1:]]}

    @staticmethod
    def _scale(line: str):
        line_split = line.split()
        scale = line_split[1].strip()
        return {'scale': float(scale)}

    @staticmethod
    def _transform(line: str):
        line_split = line.split()
        transform_value = [float(x.strip()) for x in line_split[1:]]
        transform_matrix = [transform_value[x:x+4] for x in range(0, len(transform_value), 4)]
        return {'transform': transform_matrix}
