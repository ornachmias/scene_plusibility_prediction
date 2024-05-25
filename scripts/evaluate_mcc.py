import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from experiments.models.coatnet import CoAtNet
from experiments.datasets.crt_pic_dataset import CrtPicDataset
from experiments.datasets.pic_dataset import PicDataset
from experiments.experiment_type import ExperimentType
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
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
    experiment_type = ExperimentType(7, ExperimentType.mcc)
    m = models[model_name](logger, experiment_type, output_dir, device)
    m.load_checkpoint()
    return m


def get_data_loader(model_name, model, logger):
    dataset = datasets[model_name]
    train_preprocess, val_preprocess = model.preprocess()
    experiment_type = ExperimentType(7, ExperimentType.mcc)
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
            labels = labels.clone().detach().max(1).indices
            all_labels.extend([x.cpu().item() for x in labels])
            all_preds.extend([x.cpu().item() for x in preds])

    return all_labels, all_preds


def generate_graph():
    results = {}
    for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
        results[model_name] = {}
        logger = get_logger(f'{model_name}_evaluate')
        model = get_model(model_name, logger)
        data_loader = get_data_loader(model_name, model, logger)
        labels, preds = evaluate(model, data_loader)
        print(f'Confusion matrix for model {model_name}')
        cf_matrix = confusion_matrix(labels, preds)
        print(cf_matrix)
        print()
        print(cf_matrix.diagonal()/cf_matrix.sum(axis=1))

        comparison_by_class = {}
        for label, pred in zip(labels, preds):
            if label not in comparison_by_class:
                comparison_by_class[label] = []

            comparison_by_class[label].append(int(label == pred))

        accuracy_by_class = {}
        print(model_name)
        for label in comparison_by_class:
            print(f'{label}: {sum(comparison_by_class[label])/len(comparison_by_class[label])}')
            accuracy_by_class[label] = sum(comparison_by_class[label])/len(comparison_by_class[label])

        results[model_name] = accuracy_by_class

    print()
    print(results)
    bars_colors = ['royalblue', 'seagreen', 'tomato', 'turquoise', 'pink', 'gold']
    labels = ['Plausible', 'Location', 'Rotation', 'Gravity', 'Intersection', 'Size', 'Pose']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    for i, model in enumerate(results):
        x = []
        y = []
        colors = []
        for j in range(len(labels)):
            y.append(results[model][j])
            x.append(j + (-0.2 + (0.2 * i)))
            colors.append(bars_colors[i])

        ax.bar(x, y, width=0.2, color=colors, label=model, align='center')

    ax.legend()
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_xticklabels(labels, rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_dir, 'mcc_v2.png'))


parser = argparse.ArgumentParser(description='Evaluate multi-class classification models')
parser.add_argument('--out_dir', '-o', type=new_dir, help='Path to the output directory')
parser.add_argument('--models_dir', '-m', type=existing_dir, help='Path to models output directory')
parser.add_argument('--dataset_dir', '-d', type=existing_dir, help='Path to dataset directory')
parser.add_argument('--gpu', '-g', help='GPU Id', default='0')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

graph_output_dir = args.out_dir
models_out_dir = args.models_dir
dataset_dir = args.dataset_dir
batch_size = 8

generate_graph()




