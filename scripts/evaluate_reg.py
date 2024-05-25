import argparse
import os.path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets.crt_pic_dataset import CrtPicDataset
from experiments.datasets.pic_dataset import PicDataset
from experiments.experiment_type import ExperimentType
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
from experiments.models.coatnet import CoAtNet
from experiments.models.vit import Vit
from utils.argparse_types import new_dir, existing_dir
from utils.log_utils import get_logger

models = {
    'ViT': Vit,
    'CRTNet': CrtNet,
    'ResNet': Resnet,
    'CoAtNet': CoAtNet
}

datasets = {
    'ViT': PicDataset,
    'CRTNet': CrtPicDataset,
    'ResNet': PicDataset,
    'CoAtNet': PicDataset
}


def get_model(model_name, logger):
    output_dir = os.path.join(models_out_dir, 'models')
    experiment_type = ExperimentType(1, ExperimentType.reg)
    m = models[model_name](logger, experiment_type, output_dir, device)
    m.load_checkpoint()
    return m


def get_data_loader(model_name, model, logger):
    dataset = datasets[model_name]
    train_preprocess, val_preprocess = model.preprocess()
    experiment_type = ExperimentType(1, ExperimentType.reg)
    val = dataset(logger, dataset_dir, False, val_preprocess, model.target_encoding, experiment_type, device)
    return DataLoader(val, batch_size, True)


def evaluate(model, data_loader):
    model.eval()

    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(data_loader):
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = model.predictions(outputs)
            all_labels.extend([x.cpu().item() for x in labels])
            all_preds.extend([x.cpu().item() for x in preds])

    return all_labels, all_preds


def generate_graph():
    for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
        logger = get_logger(f'{model_name}_evaluate')
        model = get_model(model_name, logger)
        data_loader = get_data_loader(model_name, model, logger)
        labels, preds = evaluate(model, data_loader)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        samples = [(y, pred) for y, pred in zip(labels, preds)]
        plausible_predictions = [x for x in preds if x > 0.9999]
        plausible_labels = [x for x in labels if x > 0.9999]
        print(f'{model_name}: '
              f'Predicted {len(plausible_predictions)/len(preds)} ({len(plausible_predictions)},{len(preds)}), '
              f'Labels: {len(plausible_labels)/len(labels)} ({len(plausible_labels)},{len(labels)})')

        sorted_by_y = sorted(samples, key=lambda tup: tup[0], reverse=False)
        n_samples = len(sorted_by_y)
        n_percentile = 5
        jump_size = n_samples / n_percentile
        for i in range(1, n_percentile + 1):
            start = int((i - 1)*jump_size)
            end = int(i*jump_size)
            diff = sorted_by_y[start:end]
            diff = [abs(x1 - x2) for x1, x2 in diff]
            print(f'{model_name} L1 Loss for {start} to {end}: {sum(diff)/len(diff)}, '
                  f'Plausibility Score range: {sorted_by_y[start][0]}-{sorted_by_y[end-1][0]}')

        (n, bins, patches) = ax.hist(labels, label='Ground Truth', bins=50, alpha=0.5)
        print(f'labels: count={n}')
        print(f'labels: bins={bins}')
        (n, bins, patches) = ax.hist(preds, label='Ground Truth', bins=50, alpha=0.5)
        print(f'labels: count={n}')
        print(f'labels: bins={bins}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_dir, f'{model_name}_reg_dist_v2.png'))


parser = argparse.ArgumentParser(description='Evaluate regression models')
parser.add_argument('--out_dir', '-o', type=new_dir, help='Path to the output directory')
parser.add_argument('--models_dir', '-m', type=existing_dir, help='Path to models output directory')
parser.add_argument('--dataset_dir', '-d', type=existing_dir, help='Path to dataset directory')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph_output_dir = args.out_dir
models_out_dir = args.models_dir
dataset_dir = args.dataset_dir
batch_size = 8

generate_graph()




