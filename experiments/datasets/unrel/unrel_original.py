import os.path

from PIL import Image
from torch.utils.data import Dataset

from utils.file_utils import load_json


class UnrelOriginal(Dataset):
    def __init__(self, logger, dataset_dir, transform, target_encoding, device):
        super().__init__()
        self.logger = logger
        self.dataset_dir = dataset_dir
        self.metadata_dir = os.path.join(dataset_dir, 'metadata')
        self.transform = transform
        self.target_encoding = target_encoding
        self.device = device
        self.data = self.initialize()

    def initialize(self):
        data = []
        for metadata_file in os.listdir(self.metadata_dir):
            metadata_path = os.path.join(self.metadata_dir, metadata_file)
            metadata = load_json(metadata_path)
            if self.skip_image(metadata['category']):
                continue

            image, label = self.get_sample(metadata)
            data.append((image, label))
        return data

    def skip_image(self, category):
        return category == 'plausible'

    def get_sample(self, metadata):
        file_name = os.path.basename(metadata['image_path'])
        image = Image.open(os.path.join(self.dataset_dir, 'images', file_name))
        if image.mode in ['L', 'RGBA']:
            rgb_img = Image.new("RGB", image.size)
            rgb_img.paste(image)
            image = rgb_img

        image = self.transform(image)
        label = {'plausible': 0, 'implausible': 1}[metadata['category']]
        return image, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, category = self.data[idx]
        return image.to(self.device), self.target_encoding(category).to(self.device)

