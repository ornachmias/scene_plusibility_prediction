import os

import torch
from PIL import Image

from experiments.datasets.unrel.unrel_original import UnrelOriginal


class CrtUnrelOriginal(UnrelOriginal):
    def get_sample(self, metadata):
        file_name = os.path.basename(metadata['image_path'])
        image = Image.open(os.path.join(self.dataset_dir, 'images', file_name))
        if image.mode in ['L', 'RGBA']:
            rgb_img = Image.new("RGB", image.size)
            rgb_img.paste(image)
            image = rgb_img

        label = {'plausible': 0, 'implausible': 1}[metadata['category']]
        orig_bbox = metadata['annotations']['bbox'][0]
        crop_bbox = (orig_bbox[0] * image.width, orig_bbox[1] * image.height,
                     orig_bbox[2] * image.width, orig_bbox[3] * image.height)
        tensor_bbox = torch.tensor([orig_bbox[0], orig_bbox[1], orig_bbox[2] - orig_bbox[0], orig_bbox[3] - orig_bbox[1]])
        sample = {
            'context_image': self.transform(image),
            'target_image': self.transform(image.crop(crop_bbox)),
            'target_bbox': tensor_bbox
        }
        return sample, label

    def __getitem__(self, index):
        sample, category = self.data[index]
        cuda_sample = {}
        for k in sample:
            cuda_sample[k] = sample[k].to(self.device)

        return cuda_sample, self.target_encoding(category).to(self.device)
