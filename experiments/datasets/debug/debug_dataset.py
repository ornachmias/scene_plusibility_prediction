import os
from logging import Logger

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from utils import file_utils


class DebugDataset(Dataset):
    def __init__(self, logger: Logger, root_dir: str, is_train: bool, transform, target_encode_fn, experiment_type, device):
        self.logger = logger
        self.root_dir = root_dir
        self.device = device
        self.categories = {'cat': 0, 'dog': 1}
        self.data = []
        self.transform = transform
        self.target_encode_fn = target_encode_fn
        self.experiment_type = experiment_type
        self.initialize(is_train)

    def initialize(self, is_train):
        os.makedirs(self.root_dir, exist_ok=True)
        gz_path = os.path.join(self.root_dir, 'cat_dog.tar.gz')

        if not os.path.isfile(gz_path):
            url = 'https://nvidia.box.com/shared/static/o577zd8yp3lmxf5zhm38svrbrv45am3y.gz'
            file_utils.download(self.logger, url, gz_path)

        extracted_dir = os.path.join(self.root_dir, 'cat_dog')
        if not os.path.isdir(extracted_dir):
            file_utils.extract(self.logger, gz_path)

        data_dir = os.path.join(extracted_dir, 'train' if is_train else 'val')
        for category in os.listdir(data_dir):
            category_dir = os.path.join(data_dir, category)
            for image_name in tqdm(os.listdir(category_dir)[:50]):
                image_path = os.path.join(category_dir, image_name)
                image = Image.open(image_path)
                if image.mode == 'L':
                    rgb_img = Image.new("RGB", image.size)
                    rgb_img.paste(image)
                    image = rgb_img
                self.data.append((self.transform(image), self.categories[category]))

    def __getitem__(self, index):
        image, category = self.data[index]
        return image.to(self.device), self.target_encode_fn(category).to(self.device)

    def __len__(self):
        return len(self.data)

