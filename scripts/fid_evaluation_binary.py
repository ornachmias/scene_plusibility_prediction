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

images_paths = {}
for group in split_scenes:
    images_paths[group] = []
    for scene_id in split_scenes[group]:
        if group.startswith('plausible'):
            plausible_dir = os.path.join(images_dir, 'plausible')
            for file_name in os.listdir(plausible_dir):
                if file_name.split('.')[0] == scene_id and file_name.endswith('.png'):
                    images_paths[group].append(os.path.join(plausible_dir, file_name))
        elif group.startswith('implausible'):
            implausible_dir = os.path.join(images_dir, 'implausible')
            for implausibility_type in os.listdir(implausible_dir):
                for file_name in os.listdir(os.path.join(implausible_dir, implausibility_type)):
                    if file_name.split('.')[0] == scene_id and file_name.endswith('.png'):
                        images_paths[group].append(os.path.join(implausible_dir, implausibility_type, file_name))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

results = {'P1/P2': [], 'P1/I1': [], 'P2/I2': [], 'I1/I2': []}
for i in range(5):
    for k in images_paths:
        random.shuffle(images_paths[k])

    min_images = min([len(images_paths[x]) for x in images_paths])
    print(f'Images in each set: {min_images}')
    plausible_1 = images_paths['plausible_1'][:min_images]
    plausible_2 = images_paths['plausible_2'][:min_images]
    implausible_1 = images_paths['implausible_1'][:min_images]
    implausible_2 = images_paths['implausible_2'][:min_images]
    results['P1/P2'].append(calculate_fid_given_images(plausible_1, plausible_2))
    print(results)
    results['P1/I1'].append(calculate_fid_given_images(plausible_1, implausible_1))
    print(results)
    results['P2/I2'].append(calculate_fid_given_images(plausible_2, implausible_2))
    print(results)
    results['I1/I2'].append(calculate_fid_given_images(implausible_1, implausible_2))
    print(results)

print(results)

