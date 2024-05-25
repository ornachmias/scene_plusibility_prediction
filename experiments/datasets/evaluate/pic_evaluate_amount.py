import os
import random
from logging import Logger

from experiments.datasets.pic_dataset import PicDataset
from experiments.experiment_type import ExperimentType


class PicEvaluateAmount(PicDataset):
    def __init__(self, logger: Logger, root_dir: str, amount, transform, target_encode_fn, device):
        self.amount = amount
        super().__init__(logger, root_dir, False, transform, target_encode_fn, ExperimentType(2, ExperimentType.bc),
                         device)

    def get_implausible_samples(self, n_samples=None, n_samples_per_category=None):
        samples = []
        implausible_dir = os.path.join(self.root_dir, 'render', 'implausible')

        for implausibility_type in os.listdir(implausible_dir):
            implausibility_dir = os.path.join(implausible_dir, implausibility_type)
            metadata_files = [x for x in os.listdir(implausibility_dir) if x.endswith('.json')]
            metadata_files = [x for x in metadata_files if int(x.split('.')[-2]) == self.amount]
            random.shuffle(metadata_files)

            for metadata_file in metadata_files:
                sample = self.get_sample(os.path.join(implausible_dir, implausibility_type, metadata_file))
                if sample is not None:
                    samples.append(sample)

        return samples

    def initialize(self):
        implausible_samples = self.get_implausible_samples()
        plausible_samples = self.get_plausible_samples(n_samples=len(implausible_samples))
        self.logger.info(f'Plausible Samples: {len(plausible_samples)}, '
                         f'Implausible Samples: {len(implausible_samples)}')
        self.data.extend(plausible_samples)
        self.data.extend(implausible_samples)
        random.shuffle(self.data)
