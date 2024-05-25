import os.path
from abc import ABC, abstractmethod
from logging import Logger

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional

from experiments.experiment_type import ExperimentType


class BaseModel(nn.Module, ABC):
    def __init__(self, logger: Logger, model_name: str, input_size: int, experiment_type: ExperimentType,
                 output_dir: str, device):
        super().__init__()
        self.logger = logger
        self.device = device
        self.experiment_type = experiment_type
        self.output_dir = os.path.join(output_dir, model_name, experiment_type.name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_name = model_name
        self.input_size = input_size
        self.model = self.init_model()
        self.model = self.model.to(self.device)

    def name(self):
        return f'{self.model_name}_{self.experiment_type}'

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    def forward(self, inputs):
        return self.model(inputs)

    def predictions(self, outputs):
        if self.experiment_type.name == ExperimentType.reg:
            return outputs.clone().detach()
        else:
            return outputs.clone().detach().max(1).indices

    def criterion(self):
        if self.experiment_type.name == ExperimentType.bc:
            return nn.BCELoss()
        elif self.experiment_type.name == ExperimentType.mcc:
            return nn.CrossEntropyLoss()
        elif self.experiment_type.name == ExperimentType.reg:
            return nn.L1Loss()

        raise ValueError(f'Invalid experiment type: {self.experiment_type.name}')

    def metric(self, y_true, y_pred):
        if self.experiment_type.name == ExperimentType.reg:
            return (1 - nn.L1Loss()(y_true, y_pred).item()) * y_true.size(0)
        else:
            return sum(y_true.max(1).indices == y_pred).item()

    def preprocess(self):
        return self.base_preprocess()

    def target_encoding(self, y):
        if self.experiment_type.name == ExperimentType.reg:
            return torch.tensor(y, dtype=torch.float).unsqueeze(dim=-1)
        else:
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.long)
            return functional.one_hot(y, self.experiment_type.n_classes).to(torch.float)

    def save_checkpoint(self, epoch, max_checkpoints=5):
        os.makedirs(self.checkpoint_dir(), exist_ok=True)
        checkpoints = os.listdir(self.checkpoint_dir())
        if len(checkpoints) == max_checkpoints:
            first_checkpoint = self.checkpoint_name(min([int(x.split('_')[0]) for x in checkpoints]))
            remove_path = os.path.join(self.checkpoint_dir(), first_checkpoint)
            os.remove(remove_path)

        checkpoint_path = os.path.join(self.checkpoint_dir(), self.checkpoint_name(epoch))
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self):
        checkpoints = os.listdir(self.checkpoint_dir())
        last_checkpoint = self.checkpoint_name(max([int(x.split('_')[0]) for x in checkpoints]))
        checkpoint_path = os.path.join(self.checkpoint_dir(), last_checkpoint)
        self.logger.info(f'{self.model_name} loaded checkpoint from {checkpoint_path}')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def base_preprocess(self):
        train_preprocess = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        val_preprocess = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        return train_preprocess, val_preprocess

    def set_parameter_requires_grad(self, model):
        n_parameters = 0
        for param in model.parameters():
            param.requires_grad = True
            n_parameters += torch.prod(torch.tensor(param.shape))

        self.logger.debug(f'{self.model_name} parameters: {n_parameters}')

    def checkpoint_dir(self):
        return os.path.join(self.output_dir, 'checkpoints')

    @staticmethod
    def checkpoint_name(epoch):
        return f'{str(epoch).zfill(3)}_state_dict.chk'


