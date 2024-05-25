import argparse
import os.path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets.evaluate.crt_pic_evaluate_amount import CrtPicEvaluateAmount
from experiments.datasets.evaluate.crt_pic_evaluate_size import CrtPicEvaluateSize
from experiments.datasets.evaluate.crt_pic_evaluate_type import CrtPicEvaluateType
from experiments.datasets.evaluate.pic_evaluate_amount import PicEvaluateAmount
from experiments.datasets.evaluate.pic_evaluate_size import PicEvaluateSize
from experiments.datasets.evaluate.pic_evaluate_type import PicEvaluateType
from experiments.experiment_type import ExperimentType
from experiments.models.coatnet import CoAtNet
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
from experiments.models.vit import Vit
from utils.argparse_types import existing_dir, new_dir
from utils.log_utils import get_logger

models = {
    'ViT': Vit,
    'CRTNet': CrtNet,
    'ResNet': Resnet,
    'CoAtNet': CoAtNet
}

datasets = {
    'ViT': {'types': PicEvaluateType, 'sizes': PicEvaluateSize, 'amount': PicEvaluateAmount},
    'CRTNet': {'types': CrtPicEvaluateType, 'sizes': CrtPicEvaluateSize, 'amount': CrtPicEvaluateAmount},
    'ResNet': {'types': PicEvaluateType, 'sizes': PicEvaluateSize, 'amount': PicEvaluateAmount},
    'CoAtNet': {'types': PicEvaluateType, 'sizes': PicEvaluateSize, 'amount': PicEvaluateAmount}
}


def get_model(model_name, logger):
    output_dir = os.path.join(models_out_dir, 'models')
    experiment_type = ExperimentType(2, ExperimentType.bc)
    m = models[model_name](logger, experiment_type, output_dir, device)
    m.load_checkpoint()
    return m


def get_data_loader(evaluation_type, current_category, model_name, model, logger):
    dataset = datasets[model_name][evaluation_type]
    train_preprocess, val_preprocess = model.preprocess()
    val = dataset(logger, dataset_dir, current_category, val_preprocess, model.target_encoding, device)
    return DataLoader(val, batch_size, True)


def evaluate(model, data_loader):
    model.eval()
    running_metric = 0

    for inputs, labels in tqdm(data_loader):
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = model.predictions(outputs)

        running_metric += model.metric(labels, preds)

    metric = running_metric / len(data_loader.dataset)
    return metric


def generate_graph(evaluation_type, categories, output_name, labels):
    metrics = {}
    for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
        logger = get_logger(f'{model_name}_evaluate')
        model = get_model(model_name, logger)
        model_metrics = {}
        for category in categories:
            data_loader = get_data_loader(evaluation_type, category, model_name, model, logger)
            type_metric = evaluate(model, data_loader)
            model_metrics[category] = type_metric

        metrics[model_name] = model_metrics

    bars_colors = ['royalblue', 'seagreen', 'tomato', 'turquoise', 'pink', 'gold']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    for i, model in enumerate(metrics):
        x = []
        y = []
        colors = []
        for j, category in enumerate(categories):
            y.append(metrics[model][category])
            x.append(j + (-0.2 + (0.2 * i)))
            colors.append(bars_colors[i])

        ax.bar(x, y, width=0.2, color=colors, label=model, align='center')

    ax.legend()
    ax.set_xticks([x for x in range(len(categories))])
    ax.set_xticklabels(labels, rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_dir, output_name))


parser = argparse.ArgumentParser(description='Evaluate binary classification models')
parser.add_argument('--out_dir', '-o', type=new_dir, help='Path to the output directory')
parser.add_argument('--models_dir', '-m', type=existing_dir, help='Path to models output directory')
parser.add_argument('--dataset_dir', '-d', type=existing_dir, help='Path to dataset directory')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph_output_dir = args.out_dir
models_out_dir = args.models_dir
dataset_dir = args.dataset_dir
batch_size = 8

implausibility_types = ['gravity', 'intersection', 'co-occurrence_location', 'co-occurrence_rotation', 'size', 'pose']
graph_labels = ['Gravity', 'Intersection', 'Location', 'Rotation', 'Size', 'Pose']
generate_graph('types', implausibility_types, 'implausibility_types_v2.png', graph_labels)
sizes = ['small', 'medium', 'large']
graph_labels = ['Small', 'Medium', 'Large']
generate_graph('sizes', sizes, 'sizes_v2.png', graph_labels)
amount = [1, 2, 3, 4, 5]
generate_graph('amount', amount, 'amount_v2.png', amount)

