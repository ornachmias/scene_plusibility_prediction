import torch
from PIL import Image

from experiments.datasets.debug.crt_debug_dataset import CrtDebugDataset


class OrvitDebugDataset(CrtDebugDataset):
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
            'target_bbox': torch.cat([bbox_relative.unsqueeze(0).to(self.device),
                                      torch.zeros((9, 4), device=self.device)])
        }
        return sample, self.target_encode_fn(category).to(self.device)
