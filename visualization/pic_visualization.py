import os
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes

from utils.file_utils import load_json


class PicVisualization:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def draw_bounding_boxes(self, out_dir):
        render_dir = os.path.join(self.root_dir, 'render')
        for plausibility_type in os.listdir(render_dir):
            plausibility_dir = os.path.join(render_dir, plausibility_type)

            if plausibility_type == 'plausible':
                metadata_files = [os.path.join(plausibility_dir, x) for x
                                  in os.listdir(plausibility_dir)
                                  if x.endswith('.json')]

                for metadata_file in metadata_files:
                    self.process_metadata(metadata_file, os.path.join(out_dir, plausibility_type))
            elif plausibility_type == 'implausible':
                for implausibility_type in os.listdir(plausibility_dir):
                    metadata_files = [os.path.join(plausibility_dir, implausibility_type, x) for x
                                      in os.listdir(os.path.join(plausibility_dir, implausibility_type))
                                      if x.endswith('.json')]

                    current_out_dir = os.path.join(out_dir, plausibility_type, implausibility_type)
                    for metadata_file in metadata_files:
                        self.process_metadata(metadata_file, current_out_dir)

    def process_metadata(self, metadata_path, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        metadata = load_json(metadata_path)
        input_image = metadata['image_path']
        output_image = os.path.basename(input_image)
        output_image = os.path.join(out_dir, output_image)
        self.draw_bounding_box(input_image, output_image, list(metadata['bbox_data'].values()))

    @staticmethod
    def draw_bounding_box(input_image, output_image, bboxes):
        img = read_image(input_image, ImageReadMode.RGB)
        boxes = torch.tensor(bboxes)
        img = draw_bounding_boxes(img, boxes, width=2)
        img = torchvision.transforms.ToPILImage()(img)
        img.save(output_image)

