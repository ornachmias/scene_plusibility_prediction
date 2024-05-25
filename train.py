import argparse
import os.path

import matplotlib

from experiments.models.coatnet import CoAtNet

matplotlib.use('Agg')

import torch

from torch.utils.data import DataLoader

from experiments.datasets.crt_pic_dataset import CrtPicDataset
from experiments.datasets.debug.crt_debug_dataset import CrtDebugDataset
from experiments.datasets.debug.debug_dataset import DebugDataset
from experiments.datasets.pic_dataset import PicDataset
from experiments.experiment_type import ExperimentType
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
from experiments.models.vit import Vit
from experiments.trainer import Trainer
from utils.file_utils import load_json
from utils.log_utils import get_logger

models = {
    'vit': Vit,
    'crt_net': CrtNet,
    'resnet': Resnet,
    'coatnet': CoAtNet
}

datasets = {
    'vit': {'debug': DebugDataset, 'pic': PicDataset},
    'coatnet': {'debug': DebugDataset, 'pic': PicDataset},
    'crt_net': {'debug': CrtDebugDataset, 'pic': CrtPicDataset},
    'resnet': {'debug': DebugDataset, 'pic': PicDataset}
}


def get_model():
    model_name = experiment_config['model']
    task_name = experiment_config['task']
    n_classes = experiment_config['n_classes']
    output_dir = os.path.join(base_config['output_dir'], 'models')

    if task_name == 'binary_classification':
        experiment_type = ExperimentType(2, ExperimentType.bc)
    elif task_name == 'classification':
        experiment_type = ExperimentType(n_classes, ExperimentType.mcc)
    elif task_name == 'regression':
        experiment_type = ExperimentType(1, ExperimentType.reg)
    else:
        raise ValueError(f'Invalid task selected: {task_name}')

    return models[model_name](logger, experiment_type, output_dir, device)


def get_data_loaders():
    model_name = experiment_config['model']
    dataset_name = experiment_config['dataset']
    dataset_dir = experiment_config['dataset_dir']
    batch_size = experiment_config['batch_size'] if 'batch_size' in experiment_config else base_config['batch_size']
    dataset = datasets[model_name][dataset_name]
    train_preprocess, val_preprocess = model.preprocess()
    train = dataset(logger, dataset_dir, True, train_preprocess, model.target_encoding, model.experiment_type, device)
    val = dataset(logger, dataset_dir, False, val_preprocess, model.target_encoding, model.experiment_type, device)
    return DataLoader(train, batch_size, True), DataLoader(val, batch_size, True)


parser = argparse.ArgumentParser(description='Run set of experiments')
parser.add_argument('--config', '-c', help='Path to the configuration file')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = load_json(args.config)
base_config = config['base_config']

for experiment_config in config['experiments']:
    logger = get_logger(f'{experiment_config["model"]}_{experiment_config["task"]}')
    model = get_model()
    train_loader, val_loader = get_data_loaders()
    trainer = Trainer(logger, model, device, base_config['early_stop'])
    trainer.fit(train_loader, val_loader, base_config['epochs'])
