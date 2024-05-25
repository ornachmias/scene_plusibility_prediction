import os
import random

import torch
from PIL import Image

from experiments.datasets.pic_dataset import PicDataset
from experiments.experiment_type import ExperimentType
from utils import math_utils
from utils.file_utils import load_json


class CrtPicDataset(PicDataset):
    def get_sample(self, metadata_path):
        scene_id, image_path, label, bbox = self.parse_metadata(metadata_path)
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
            return sample, label

        return None

    def parse_metadata(self, metadata_path):
        metadata = load_json(metadata_path)
        is_plausible = 'transformations' not in metadata or len(metadata['transformations']) == 0
        bboxes = metadata['bbox_data']

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
            return metadata['scene_id'], metadata['image_path'], int(is_plausible), bbox
        elif self.exp_type.name == ExperimentType.mcc:
            if is_plausible:
                return metadata['scene_id'], metadata['image_path'], self.categories['plausible'], bbox

            label = self.categories[metadata['transformations'][0]['transformation_type']]
            return metadata['scene_id'], metadata['image_path'], label, bbox
        elif self.exp_type.name == ExperimentType.reg:
            return metadata['scene_id'], metadata['image_path'], self.regression_score(metadata), bbox

        return None

    def __getitem__(self, index):
        sample, category = self.data[index]
        cuda_sample = {}
        for k in sample:
            cuda_sample[k] = sample[k].to(self.device)

        return cuda_sample, self.target_encode_fn(category).to(self.device)

