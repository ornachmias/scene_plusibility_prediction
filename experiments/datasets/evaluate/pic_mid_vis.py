import os.path
import random
from logging import Logger

from PIL import Image
from shapely.ops import unary_union
from torch.utils.data import Dataset
from tqdm import tqdm
from shapely.geometry import box

from experiments.experiment_type import ExperimentType
from utils import math_utils
from utils.file_utils import load_json


class PicMidVis(Dataset):
    def __init__(self, logger: Logger, root_dir: str, is_train: bool, transform, target_encode_fn,
                 exp_type: ExperimentType, device):
        self.logger = logger
        self.device = device
        self.root_dir = root_dir
        self.exp_type = exp_type
        self.categories = self.get_categories()
        scenes_split = load_json('./resources/pic/dataset_split.json')
        self.scenes = scenes_split['train'] if is_train else scenes_split['val']
        self.data = []
        self.transform = transform
        self.target_encode_fn = target_encode_fn
        self.original_image_size = 512
        self.initialize()
        self.logger.info(f'Pic Dataset ({"train" if is_train else "val"}): {len(self.data)} samples, '
                         f'{len(self.categories) if self.categories else 1} categories')

    def initialize(self):
        samples = []
        implausible_dir = os.path.join(self.root_dir, 'render', 'implausible')
        samples_names = ['co-occurrence_location/termFigE.generated_camera_108.0002.json',
                         'co-occurrence_rotation/ZachAngie.generated_camera_037.0003.json',
                         'gravity/DiningCasual0.generated_camera_044.0002.json',
                         'intersection/GaborGates381.generated_camera_093.0002.json',
                         'pose/DiningFormal1.generated_camera_047.0004.json',
                         'size/ZachDining2.generated_camera_064.0001.json']
        for samples_name in samples_names:
            sample = self.get_sample(os.path.join(implausible_dir, samples_name))
            samples.append(sample)

        self.data = samples

    def parse_metadata(self, metadata_path):
        metadata = load_json(metadata_path)
        is_plausible = 'transformations' not in metadata or len(metadata['transformations']) == 0

        if self.exp_type.name == ExperimentType.bc:
            return metadata['scene_id'], metadata['image_path'], int(is_plausible)
        elif self.exp_type.name == ExperimentType.mcc:
            if is_plausible:
                return metadata['scene_id'], metadata['image_path'], self.categories['plausible']

            label = self.categories[metadata['transformations'][0]['transformation_type']]
            return metadata['scene_id'], metadata['image_path'], label
        elif self.exp_type.name == ExperimentType.reg:
            return metadata['scene_id'], metadata['image_path'], self.regression_score(metadata)

        return None

    def regression_score(self, metadata):
        if 'transformations' not in metadata:
            return 1

        image_size = self.original_image_size

        bboxes = []
        for transformation in metadata['transformations']:
            obj_name = transformation['obj_name']
            if obj_name in metadata['bbox_data']:
                bbox = math_utils.clamp_bbox(metadata['bbox_data'][obj_name], image_size)
                if math_utils.bbox_valid(bbox):
                    bboxes.append(bbox)

        if len(bboxes) == 0:
            return 1.

        bboxes = [box(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bboxes]
        u = unary_union(bboxes).area
        return 1 - (u / (image_size ** 2))

    def get_categories(self):
        if self.exp_type.name == ExperimentType.mcc:
            return {'plausible': 0, 'co-occurrence_location': 1, 'co-occurrence_rotation': 2, 'gravity': 3,
                    'intersection': 4, 'size': 5, 'pose': 6}
        elif self.exp_type.name == ExperimentType.bc:
            return {'plausible': 0, 'implausible': 1}

        return None

    def get_sample(self, metadata_path):
        scene_id, image_path, label = self.parse_metadata(metadata_path)
        if scene_id in self.scenes and os.path.isfile(image_path):
            image = Image.open(image_path)
            if image.mode in ['L', 'RGBA']:
                rgb_img = Image.new("RGB", image.size)
                rgb_img.paste(image)
                image = rgb_img

            image = self.transform(image)
            return image, label

        return None

    def __getitem__(self, index):
        image, category = self.data[index]
        return image.to(self.device), self.target_encode_fn(category).to(self.device)

    def __len__(self):
        return len(self.data)
