from logging import Logger

import torchvision
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
from torchvision.models import ResNet101_Weights

from experiments.experiment_type import ExperimentType
from experiments.models.base_model import BaseModel


class Resnet(BaseModel):
    def __init__(self, logger: Logger, experiment_type: ExperimentType, output_dir: str, device):
        super().__init__(logger, 'ResNet101', 224, experiment_type, output_dir, device)

    def init_model(self):
        model_ft = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, self.experiment_type.n_classes)
        self.set_parameter_requires_grad(model_ft)
        return model_ft

    def optimizer(self):
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    def criterion(self):
        if self.experiment_type.name == ExperimentType.bc:
            return BCEWithLogitsLoss()
        else:
            return super().criterion()



