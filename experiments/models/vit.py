from logging import Logger

import timm
from torch import optim
from torch.nn import BCEWithLogitsLoss

from experiments.experiment_type import ExperimentType
from experiments.models.base_model import BaseModel


class Vit(BaseModel):

    def __init__(self, logger: Logger, experiment_type: ExperimentType, output_dir: str, device):
        super().__init__(logger, 'ViT', 224, experiment_type, output_dir, device)

    def optimizer(self):
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    def init_model(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.experiment_type.n_classes)
        self.set_parameter_requires_grad(model)
        return model

    def criterion(self):
        if self.experiment_type.name == ExperimentType.bc:
            return BCEWithLogitsLoss()
        else:
            return super().criterion()


