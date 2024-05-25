import os
import random
from logging import Logger

from tqdm import tqdm

from experiments.datasets.crt_pic_dataset import CrtPicDataset
from experiments.experiment_type import ExperimentType
from utils.file_utils import load_json


class CrtPicEvaluateSize(CrtPicDataset):
    def __init__(self, logger: Logger, root_dir: str, size, transform, target_encode_fn, device):
        self.size = size
        self.sizes_by_label = load_json('./resources/pic/models_size.json')
        super().__init__(logger, root_dir, False, transform, target_encode_fn, ExperimentType(2, ExperimentType.bc),
                         device)

    def get_implausible_samples(self, n_samples=None, n_samples_per_category=None):
        samples = []
        implausible_dir = os.path.join(self.root_dir, 'render', 'implausible')

        for implausibility_type in tqdm(os.listdir(implausible_dir), desc='Implausible Samples'):
            category_samples = []
            implausibility_dir = os.path.join(implausible_dir, implausibility_type)
            metadata_files = [x for x in os.listdir(implausibility_dir) if x.endswith('.json')]
            random.shuffle(metadata_files)

            for metadata_file in metadata_files:
                metadata_path = os.path.join(implausible_dir, implausibility_type, metadata_file)
                metadata = load_json(metadata_path)
                if 'transformations' in metadata and len(metadata['transformations']) != 1:
                    continue

                obj_name = metadata['transformations'][0]['obj_name']
                obj_category = [x['category'] for x in metadata['objects_metadata'] if x['name'] == obj_name]
                if not obj_category:
                    continue

                obj_category = obj_category[0]

                if self.sizes_by_label[obj_category] != self.size:
                    continue

                sample = self.get_sample(metadata_path)
                if sample is not None:
                    category_samples.append(sample)

                if n_samples_per_category is not None and len(category_samples) >= n_samples_per_category:
                    break

            samples.extend(category_samples)

        random.shuffle(samples)
        if n_samples is not None:
            samples = samples[:n_samples]

        return samples

    def initialize(self):
        implausible_samples = self.get_implausible_samples()
        plausible_samples = self.get_plausible_samples(n_samples=len(implausible_samples))
        self.logger.info(f'Plausible Samples: {len(plausible_samples)}, '
                         f'Implausible Samples: {len(implausible_samples)}')
        self.data.extend(plausible_samples)
        self.data.extend(implausible_samples)
        random.shuffle(self.data)