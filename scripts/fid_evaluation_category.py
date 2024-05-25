import argparse
import os
import random

import torch

from experiments.fid_inception import InceptionV3, calculate_frechet_distance, calculate_activation_statistics
from utils.argparse_types import existing_dir, existing_file
from utils.file_utils import load_json


# Code by https://github.com/mseitzer/pytorch-fid
# Copied manually due to lack support of Python API


def calculate_fid_given_images(images_paths_1, images_paths_2, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(images_paths_1, model, device=device)
    m2, s2 = calculate_activation_statistics(images_paths_2, model, device=device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


parser = argparse.ArgumentParser(description='Run calculation of FID score')
parser.add_argument('--renders_dir', '-r', type=existing_dir, help='Path to dataset directory')
parser.add_argument('--scenes_split', '-s', type=existing_file, help='Path to split for FID score',
                    default='./resources/pic/fid_scenes_split.json')

args = parser.parse_args()
split_scenes = load_json(args.scenes_split)
images_dir = args.renders_dir

scenes_set_1 = set(split_scenes['plausible_1'])
scenes_set_2 = set(split_scenes['implausible_1'])
implausibility_categories = ['gravity', 'intersection', 'pose', 'co-occurrence_location',
                             'co-occurrence_rotation', 'size']

image_set_1 = {x: [] for x in implausibility_categories}
image_set_2 = {x: [] for x in implausibility_categories}

# Load all images
implausible_dir = os.path.join(images_dir, 'implausible')
for implausibility_category in implausibility_categories:
    category_dir = os.path.join(implausible_dir, implausibility_category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.png'):
            scene_id = file_name.split('.')[0]
            if scene_id in scenes_set_1:
                image_set_1[implausibility_category].append(file_path)
            else:
                image_set_2[implausibility_category].append(file_path)

# Shuffle and find the minimum number of images in category
min_images = None
for image_set in [image_set_1, image_set_2]:
    for category in image_set:
        if min_images is None or len(image_set[category]) < min_images:
            min_images = len(image_set[category])

print(f'Limiting each category to {min_images} images')

# Equalize categories
for image_set in [image_set_1, image_set_2]:
    for category in image_set:
        random.shuffle(image_set[category])
        image_set[category] = image_set[category][:min_images]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

images_sets = {
    'exp_1': {
        'set_1': image_set_1,
        'set_2': image_set_2
    },
    'exp_2': {
        'set_1': image_set_2,
        'set_2': image_set_1
    }
}

results = {}
for _ in range(5):
    for exp_id in images_sets:
        for category in images_sets[exp_id]['set_1']:
            exp_key = f'{exp_id}/{category}'
            if exp_key not in results:
                results[exp_key] = []

            category_images = images_sets[exp_id]['set_1'][category]
            other_images = []
            for k in images_sets[exp_id]['set_2']:
                if k != category:
                    other_images.extend(images_sets[exp_id]['set_2'][k])

            score = calculate_fid_given_images(category_images, other_images)
            print(f'Experiment Id: {exp_key}, Score: {score}')
            results[exp_key].append(calculate_fid_given_images(category_images, other_images))

print('')
print('')
print(results)
