import argparse
import csv
import json
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets.evaluate.crt_pic_evaluate_category import CrtPicEvaluateCategory
from experiments.datasets.evaluate.pic_evaluate_category import PicEvaluateCategory
from experiments.experiment_type import ExperimentType
from experiments.models.crtnet import CrtNet
from experiments.models.resnet import Resnet
from experiments.models.coatnet import CoAtNet
from experiments.models.vit import Vit
from utils.argparse_types import existing_dir, existing_file, new_dir
from utils.file_utils import load_json
from utils.log_utils import get_logger


def get_category_from_metadata(metadata_content, model_id):
    if 'room' in model_id:
        return None

    objects_metadata = metadata_content['objects_metadata']
    for object_metadata in objects_metadata:
        if object_metadata['name'] == model_id or object_metadata['id'] == model_id:
            return object_metadata['category']

    return None


parser = argparse.ArgumentParser(description='Run experiments related to the supplemental section')
parser.add_argument('--scenes_file', '-s', type=existing_file, help='Path to the Scenes 3D metadata file')
parser.add_argument('--scenes_ann', '-n', type=existing_file, help='Path to the Scenes 3D annotation file')
parser.add_argument('--output_dir', '-o', type=new_dir, help='Path to the directory containing output files')
parser.add_argument('--models_dir', '-m', type=existing_dir, help='Path to the directory models checkpoints')
parser.add_argument('--pic_dir', '-p', type=existing_dir, help='Path to the directory models checkpoints')
args = parser.parse_args()


root_dir = os.path.join(args.pic_dir, 'render')
implausible_dir = os.path.join(root_dir, 'implausible')
plausible_dir = os.path.join(root_dir, 'plausible')

# scenes_3d_metadata = args.scenes_file
# scenes_3d_annotations = load_json(args.scenes_ann)
# with open(scenes_3d_metadata) as f:
#     scenes_metadata_lines = f.readlines()
#
# categories = {}
# for line in scenes_metadata_lines[1:]:
#     separator_index = line.find(',')
#     scene_name = line[:separator_index]
#     scene_data = line[separator_index + 1:]
#     scene_data = json.loads(scene_data[1:-2])
#     models = [x['modelID'] for x in scene_data['objects']]
#     for model in models:
#         if model not in scenes_3d_annotations or 'room' in model:
#             continue
#
#         annotation = scenes_3d_annotations[model]
#         if 'sub_category' in annotation:
#             category = annotation['sub_category']
#         else:
#             category = annotation['category']
#
#         if category not in categories:
#             categories[category] = {'general_scenes': 0}
#
#         categories[category]['general_scenes'] += 1
#
#
# for file_name in os.listdir(plausible_dir):
#     if not file_name.endswith('.json'):
#         continue
#
#     metadata = os.path.join(plausible_dir, file_name)
#     metadata = load_json(metadata)
#     for visible_object in metadata['visible_objects']:
#         category = get_category_from_metadata(metadata, visible_object)
#         if category is None:
#             continue
#
#         if 'plausible_images' not in categories[category]:
#             categories[category]['plausible_images'] = 0
#
#         categories[category]['plausible_images'] += 1
#
# implausibility_types = os.listdir(implausible_dir)
# for implausibility_type in implausibility_types:
#     implausibility_dir = os.path.join(implausible_dir, implausibility_type)
#     for file_name in os.listdir(implausibility_dir):
#         if not file_name.endswith('.json'):
#             continue
#
#         metadata = os.path.join(implausibility_dir, file_name)
#         metadata = load_json(metadata)
#         for visible_object in metadata['visible_objects']:
#             category = get_category_from_metadata(metadata, visible_object)
#             if category is None:
#                 continue
#
#             if 'implausible_images' not in categories[category]:
#                 categories[category]['implausible_images'] = 0
#
#             categories[category]['implausible_images'] += 1
#
#         if 'transformations' in metadata and len(metadata['transformations']) > 0:
#             for transformation in metadata['transformations']:
#                 category = get_category_from_metadata(metadata, transformation['obj_name'])
#                 if category is None:
#                     continue
#
#                 if f'transformation_{implausibility_type}' not in categories[category]:
#                     categories[category][f'transformation_{implausibility_type}'] = 0
#
#                 categories[category][f'transformation_{implausibility_type}'] += 1
#
#                 if 'transformations' not in categories[category]:
#                     categories[category]['transformations'] = 0
#
#                 categories[category]['transformations'] += 1
#
# fieldnames = set()
# for category in categories:
#     for k in categories[category]:
#         fieldnames.add(k)
#
# fieldnames.add('category')
# fieldnames = sorted(list(fieldnames))
# csv_rows = []
# for category in categories:
#     row = {'category': category}
#     for k in categories[category]:
#         row[k] = categories[category][k]
#
#     csv_rows.append(row)
#
# dist_fields = set()
# for field in fieldnames:
#     if field == 'category':
#         continue
#
#     field_sum = sum([x[field] for x in csv_rows if field in x])
#     for row in csv_rows:
#         dist_fields.add(f'{field}_dist')
#         if field not in row:
#             row[field] = 0
#
#         row[f'{field}_dist'] = f'{(row[field] / field_sum):.3f}'
#
# csv_fields = fieldnames.copy()
# csv_fields.extend(list(dist_fields))
# output_file = os.path.join(args.output_dir, 'dataset_dist.csv')
# with open(output_file, 'w', encoding='UTF8', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=csv_fields)
#     writer.writeheader()
#     writer.writerows(csv_rows)


models = {
    'ViT': Vit,
    'CRTNet': CrtNet,
    'ResNet': Resnet,
    'CoAtNet': CoAtNet
}

datasets = {
    'ViT': PicEvaluateCategory,
    'CRTNet': CrtPicEvaluateCategory,
    'ResNet': PicEvaluateCategory,
    'CoAtNet': PicEvaluateCategory
}

results = {}


def evaluate(model, data_loader, dataset):
    cf = {}
    metric = {}
    model.eval()
    n_samples = 0
    for inputs, indices, labels in tqdm(data_loader):
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = model.predictions(outputs)
            indices = [x.item() for x in indices]
            labels = labels.max(1).indices
            for i, index in enumerate(indices):
                n_samples += 1
                categories = dataset.data[index][2]
                pred = preds[i].item()
                label = labels[i].item()
                for category in categories:
                    if label not in metric:
                        metric[label] = {}

                    if category not in metric[label]:
                        metric[label][category] = []

                    if category not in cf:
                        cf[category] = {}

                    if pred not in cf[category]:
                        cf[category][pred] = 0

                    cf[category][pred] += 1
                    metric[label][category].append(int(pred == label))

    print(f'{model.model_name} predictions count by category:')
    print(json.dumps(cf))

    return metric, n_samples


def get_model(model_name, logger, experiment_type):
    output_dir = os.path.join(args.models_dir, 'models')
    m = models[model_name](logger, experiment_type, output_dir, device)
    m.load_checkpoint()
    return m


def get_data_loader(model_name, model, logger, experiment_type):
    dataset = datasets[model_name]
    train_preprocess, val_preprocess = model.preprocess()
    val = dataset(logger, dataset_dir, False, val_preprocess, model.target_encoding, experiment_type, device)
    return DataLoader(val, batch_size, True), val


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dataset_dir = args.pic_dir
batch_size = 8

models_metric = {}
exp_type = ExperimentType(2, ExperimentType.bc)
for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
    logger = get_logger(f'{model_name}_evaluate')
    model = get_model(model_name, logger, exp_type)
    data_loader, dataset = get_data_loader(model_name, model, logger, exp_type)
    current_metric, n_samples = evaluate(model, data_loader, dataset)
    for label in current_metric:
        model_key = f'{model_name}_bc_{label}'

        for category in current_metric[label]:
            if category not in models_metric:
                models_metric[category] = {}

            models_metric[category][model_key] = sum(current_metric[label][category]) / len(current_metric[label][category])

exp_type = ExperimentType(7, ExperimentType.mcc)
for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
    logger = get_logger(f'{model_name}_evaluate')
    model = get_model(model_name, logger, exp_type)
    data_loader, dataset = get_data_loader(model_name, model, logger, exp_type)
    current_metric, n_samples = evaluate(model, data_loader, dataset)
    for label in current_metric:
        model_key = f'{model_name}_mcc_{label}'

        for category in current_metric[label]:
            if category not in models_metric:
                models_metric[category] = {}

            models_metric[category][model_key] = sum(current_metric[label][category]) / len(current_metric[label][category])

csv_rows = []
fieldnames = set()
for category in models_metric:
    fieldnames.add('category')
    row = {'category': category}
    for model_key in models_metric[category]:
        fieldnames.add(model_key)
        row[model_key] = models_metric[category][model_key]

    csv_rows.append(row)


output_file = os.path.join(args.output_dir, 'model_preds_dist_v2.csv')
with open(output_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

