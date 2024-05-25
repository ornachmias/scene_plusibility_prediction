import argparse
import os

from utils.argparse_types import existing_dir

parser = argparse.ArgumentParser(description='Count the number of images generated for each scene')
parser.add_argument('--renders_dir', '-r', type=existing_dir, help='Path to the renders directory')
args = parser.parse_args()


root_dir = args.renders_dir
plausible_dir = os.path.join(root_dir, 'plausible')
implausible_dir = os.path.join(root_dir, 'implausible')

scenes_count = {}

for file_name in os.listdir(plausible_dir):
    if not file_name.endswith('.png'):
        continue

    scene_id = file_name.split('.')[0]
    if scene_id not in scenes_count:
        scenes_count[scene_id] = {}

    if 'plausible' not in scenes_count[scene_id]:
        scenes_count[scene_id]['plausible'] = 0

    scenes_count[scene_id]['plausible'] += 1

for plausibility_type in os.listdir(implausible_dir):
    for file_name in os.listdir(os.path.join(implausible_dir, plausibility_type)):
        if not file_name.endswith('.png'):
            continue

        scene_id = file_name.split('.')[0]
        if scene_id not in scenes_count:
            scenes_count[scene_id] = {}

        if plausibility_type not in scenes_count[scene_id]:
            scenes_count[scene_id][plausibility_type] = 0

        scenes_count[scene_id][plausibility_type] += 1


for scene_id in scenes_count:
    total = sum(scenes_count[scene_id].values())
    print(f'Scene {scene_id} Total Count: {total}')
    print('=======================================')
    for plausibility_type in scenes_count[scene_id]:
        print(f'{plausibility_type}={scenes_count[scene_id][plausibility_type]}')

    print('')
