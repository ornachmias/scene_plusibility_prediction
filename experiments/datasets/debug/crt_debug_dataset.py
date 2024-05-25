import os.path
from logging import Logger

import torch
from torch.utils.data import Dataset
from xml.etree import ElementTree
from PIL import Image

from utils import file_utils


class CrtDebugDataset(Dataset):
    def __init__(self, logger: Logger, root_dir: str, is_train: bool, transform, target_encode_fn, device):
        self.logger = logger
        self.root_dir = root_dir
        self.device = device
        self.transform = transform
        self.categories = {'cat': 0, 'dog': 1}
        self.target_encode_fn = target_encode_fn
        self.data = []
        self.initialize(is_train)

    def initialize(self, is_train):
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, 'archive.zip')
        if not os.path.isfile(zip_path):
            url = 'https://drive.google.com/u/0/uc?id=1cwTcG76Pp2mi2QVX3jQ8OdMEGsnNVRpF&export=download'
            file_utils.google_drive_download(url, zip_path)

        annotations_dir = os.path.join(self.root_dir, 'annotations')
        if not os.path.isdir(annotations_dir) or not os.path.isdir(os.path.join(self.root_dir, 'images')):
            file_utils.extract(self.logger, zip_path)

        for annotation_name in os.listdir(annotations_dir):
            i = int(annotation_name.replace('Cats_Test', '').replace('.xml', ''))
            if i > 3000 and is_train:
                continue

            annotation_path = os.path.join(annotations_dir, annotation_name)
            root = ElementTree.parse(annotation_path).getroot()
            image_path = os.path.join(self.root_dir, root.find('folder').text, root.find('filename').text)
            category = self.categories[root.find('object/name').text]
            root_bbox = root.find('object/bndbox')
            bbox = []
            for bbox_point in ['xmin', 'ymin', 'xmax', 'ymax']:
                bbox.append(int(root_bbox.find(bbox_point).text))

            self.data.append((image_path, category, bbox))

    def __getitem__(self, index):
        image_path, category, bbox = self.data[index]
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
            'context_image': self.transform(image).to(self.device),
            'target_image': self.transform(image.crop(bbox)).to(self.device),
            'target_bbox': bbox_relative.to(self.device)
        }
        return sample, self.target_encode_fn(category).to(self.device)

    def __len__(self):
        return len(self.data)
