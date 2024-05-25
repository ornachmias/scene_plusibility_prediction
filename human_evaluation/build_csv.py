import argparse
import csv
import os
import random

parser = argparse.ArgumentParser(description='Build CSV file containing HIT entries for AMT')
parser.add_argument('--url', '-u', help='Base url for files')
parser.add_argument('--examples', '-e', help='Base url for examples')
parser.add_argument('--output_dir', '-o', help='Path to a directory that will contain the CSV file')
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'hit_entries.csv')
url_base = args.url
examples_url = args.examples
n_plausible = 15
n_category_implausible = 15
n_image_appearances = 3
n_image_per_hit = 5

test_images = [
    (examples_url + 'intersection_example.png', 'implausible_intersection'),
    (examples_url + 'gravity_example.png', 'implausible_gravity'),
    (examples_url + 'location_example.png', 'implausible_co-occurrence_location'),
    (examples_url + 'pose_example.png', 'implausible_pose'),
    (examples_url + 'rotation_example.png', 'implausible_co-occurrence_rotation'),
    (examples_url + 'size_example.png', 'implausible_size'),
]

render_dir = './data/datasets/pic/render'
plausible_dir = os.path.join(render_dir, 'plausible')
implausible_dir = os.path.join(render_dir, 'implausible')

plausible_images = ['/'.join(['plausible', x]) for x in os.listdir(plausible_dir) if x.endswith('.png')]
plausible_images = random.sample(plausible_images, k=n_plausible)
plausible_images = [{'image_path': x, 'category': 'plausible'} for x in plausible_images]

implausible_images = []
for category in os.listdir(implausible_dir):
    category_dir = os.path.join(implausible_dir, category)
    if not os.path.isdir(category_dir):
        continue

    category_images = ['/'.join(['implausible', category, x]) for x in os.listdir(category_dir) if x.endswith('.png')]
    category_images = random.sample(category_images, k=n_category_implausible)
    category_images = [{'image_path': x, 'category': category} for x in category_images]
    implausible_images.extend(category_images)

images_counter = {}
all_images = {}

for i in plausible_images:
    all_images[i['image_path']] = i

for i in implausible_images:
    all_images[i['image_path']] = i

csv_lines = []
while len(all_images) > 0:
    sampled_images = random.sample(list(all_images.keys()), min(n_image_per_hit, len(all_images)))
    csv_line = {}
    for i, s in enumerate(sampled_images):
        csv_line[f'image{i + 1}_url'] = url_base + all_images[s]['image_path']
        csv_line[f'image{i + 1}_category'] = all_images[s]['category']

        if s not in images_counter:
            images_counter[s] = 0

        images_counter[s] += 1

    test_sample = random.choice(test_images)
    csv_line['image_test_url'] = test_sample[0]
    csv_line['image_test_result'] = test_sample[1]

    csv_lines.append(csv_line)
    remove_images = [x for x in sampled_images if images_counter[x] == n_image_appearances]
    for remove_image in remove_images:
        all_images.pop(remove_image)

csv_lines = [x for x in csv_lines if len(x) == 2 * (n_image_per_hit + 1)]
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_lines[0].keys())
    writer.writeheader()
    writer.writerows(csv_lines)






