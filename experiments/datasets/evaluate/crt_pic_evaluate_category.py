import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from experiments.datasets.crt_pic_dataset import CrtPicDataset
from experiments.experiment_type import ExperimentType
from utils import math_utils
from utils.file_utils import load_json


class CrtPicEvaluateCategory(CrtPicDataset):
    def get_implausible_samples(self, n_samples=None, n_samples_per_category=None):
        samples = []
        implausible_dir = os.path.join(self.root_dir, 'render', 'implausible')
        plausible_dir = os.path.join(self.root_dir, 'render', 'plausible')

        for implausibility_type in tqdm(os.listdir(implausible_dir), desc='Implausible Samples'):
            category_samples = []
            implausibility_dir = os.path.join(implausible_dir, implausibility_type)
            metadata_files = [x for x in os.listdir(implausibility_dir) if x.endswith('.json')]
            random.shuffle(metadata_files)

            for metadata_file in metadata_files:
                sample = self.get_sample(os.path.join(implausible_dir, implausibility_type, metadata_file))
                plausible_metadata = os.path.join(plausible_dir, '.'.join(metadata_file.split('.')[:2]) + '.0000.json')

                plausible_sample = self.get_sample(plausible_metadata)
                if plausible_sample is not None and sample is not None:
                    plausible_sample = list(plausible_sample)
                    plausible_sample[2] = sample[2]
                    plausible_sample = tuple(plausible_sample)
                    category_samples.append(plausible_sample)

                if sample is not None:
                    category_samples.append(sample)

                if n_samples_per_category is not None and len(category_samples) >= n_samples_per_category:
                    break

            samples.extend(category_samples)

        random.shuffle(samples)
        if n_samples is not None:
            samples = samples[:n_samples]

        return samples

    def get_sample(self, metadata_path):
        scene_id, image_path, label, bbox, categories = self.parse_metadata(metadata_path)
        if bbox is None:
            return None

        if scene_id in self.scenes and os.path.isfile(image_path):
            image = Image.open(image_path)
            if image.mode in ['L', 'RGBA']:
                rgb_img = Image.new("RGB", image.size)
                rgb_img.paste(image)
                image = rgb_img

            bbox_relative = torch.tensor([bbox[0] / image.width,
                                          bbox[1] / image.height,
                                          (bbox[2] - bbox[0]) / image.width,
                                          (bbox[3] - bbox[1]) / image.height])
            sample = {
                'context_image': self.transform(image),
                'target_image': self.transform(image.crop(bbox)),
                'target_bbox': bbox_relative
            }
            return sample, label, categories

        return None

    def parse_metadata(self, metadata_path):
        metadata = load_json(metadata_path)
        is_plausible = 'transformations' not in metadata or len(metadata['transformations']) == 0
        bboxes = metadata['bbox_data']
        categories = []
        if not is_plausible and 'transformations' in metadata:
            labels = {x['name']: x['category'] for x in metadata['objects_metadata']}

            for transformation in metadata['transformations']:
                obj_name = transformation['obj_name']
                if obj_name in labels:
                    categories.append(labels[obj_name])

        if is_plausible:
            bbox_keys = list(bboxes.keys())
            random.shuffle(bbox_keys)
        else:
            bbox_keys = list([x['obj_name'] for x in metadata['transformations']])
            random.shuffle(bbox_keys)

        bbox = None
        for bbox_key in bbox_keys:
            if bbox_key not in bboxes:
                continue

            bbox = bboxes[bbox_key]
            bbox = math_utils.clamp_bbox(bbox, self.original_image_size)
            if math_utils.bbox_valid(bbox):
                break

        if not math_utils.bbox_valid(bbox):
            bbox = None

        if self.exp_type.name == ExperimentType.bc:
            return metadata['scene_id'], metadata['image_path'], int(is_plausible), bbox, categories
        elif self.exp_type.name == ExperimentType.mcc:
            if is_plausible:
                return metadata['scene_id'], metadata['image_path'], self.categories['plausible'], bbox, categories

            label = self.categories[metadata['transformations'][0]['transformation_type']]
            return metadata['scene_id'], metadata['image_path'], label, bbox, categories

        return None

    def initialize(self):
        implausible_samples = self.get_implausible_samples()
        self.logger.info(f'Plausible Samples: {len([x for x in implausible_samples if x[1] == 0])}, '
                         f'Implausible Samples: {len([x for x in implausible_samples if x[1] == 1])}')
        self.data.extend(implausible_samples)
        random.shuffle(self.data)

    def __getitem__(self, index):
        sample, category, categories = self.data[index]
        cuda_sample = {}
        for k in sample:
            cuda_sample[k] = sample[k].to(self.device)

        return cuda_sample, torch.tensor(index).to(self.device), self.target_encode_fn(category).to(self.device)