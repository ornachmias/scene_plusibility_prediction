import argparse
import os.path
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets.unrel.crt_unrel_original import CrtUnrelOriginal
from experiments.datasets.unrel.unrel_original import UnrelOriginal
from experiments.experiment_type import ExperimentType
from experiments.models.coatnet import CoAtNet
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
from experiments.models.vit import Vit
from utils.argparse_types import existing_dir
from utils.log_utils import get_logger

models = {
    'ViT': Vit,
    'CRTNet': CrtNet,
    'ResNet': Resnet,
    'CoAtNet': CoAtNet
}

datasets = {
    'ViT': UnrelOriginal,
    'CRTNet': CrtUnrelOriginal,
    'ResNet': UnrelOriginal,
    'CoAtNet': UnrelOriginal
}


def get_model(model_name, logger):
    output_dir = os.path.join(models_out_dir, 'models')
    experiment_type = ExperimentType(2, ExperimentType.bc)
    m = models[model_name](logger, experiment_type, output_dir, device)
    m.load_checkpoint()
    return m


def get_data_loader(model_name, model, logger):
    dataset = datasets[model_name]
    train_preprocess, val_preprocess = model.preprocess()
    val = dataset(logger, dataset_dir, val_preprocess, model.target_encoding, device)
    return DataLoader(val, batch_size, True)


def evaluate_model(model, data_loader):
    model.eval()
    running_metric = 0

    for inputs, labels in tqdm(data_loader):
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = model.predictions(outputs)

        running_metric += model.metric(labels, preds)

    metric = running_metric / len(data_loader.dataset)
    return metric


def evaluate_all():
    metrics = {}
    for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
        logger = get_logger(f'{model_name}_evaluate_unrel')
        model = get_model(model_name, logger)
        data_loader = get_data_loader(model_name, model, logger)
        metrics[model_name] = evaluate_model(model, data_loader)

    return metrics


parser = argparse.ArgumentParser(description='Evaluate Unrel binary classification models')
parser.add_argument('--models_dir', '-m', type=existing_dir, help='Path to models output directory')
parser.add_argument('--dataset_dir', '-d', type=existing_dir, help='Path to dataset directory')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models_out_dir = args.models_dir
dataset_dir = args.dataset_dir
batch_size = 4

result = evaluate_all()
print(result)
