import os
import argparse
import random
import shutil

import requests
from PIL import Image
from pycocotools.coco import COCO
from scipy.io import loadmat
from tqdm import tqdm

from utils.argparse_types import existing_dir, new_dir, existing_file
from utils.file_utils import save_json, load_json


def load_annotations(annotations_path, images_dir, dest_images_dir):
    all_categories = set()
    result = []
    mat = loadmat(annotations_path, struct_as_record=False, squeeze_me=True)
    for annotation in mat['annotations']:
        file_name = annotation.filename
        orig_image_path = os.path.join(images_dir, file_name)
        image = Image.open(orig_image_path)
        bboxs = []
        categories = []
        for obj in annotation.objects:
            all_categories.add(obj.category)
            categories.append(obj.category)
            bbox = obj.box
            formatted_bbox = [bbox[0] / image.size[0], bbox[1] / image.size[1],
                              bbox[2] / image.size[0], bbox[3] / image.size[1]]
            bboxs.append(formatted_bbox)

        result.append({'bbox': bboxs, 'category': categories,
                       'file_path': os.path.join(dest_images_dir, file_name)})
        shutil.copy(orig_image_path, os.path.join(dest_images_dir, file_name))

    return result


def find_mscoco_candidates(categories, coco_dataset, images_output_dir):
    n_categories = len(categories)
    while True:
        categories_dict = {}
        for cat in categories[:n_categories]:
            coco_cat_id = coco_dataset.getCatIds(catNms=[unrel_to_ms_coco_categories[cat]])
            if coco_cat_id:
                categories_dict[coco_cat_id[0]] = unrel_to_ms_coco_categories[cat]
            else:
                not_found_categories.add(cat)

        image_ids = coco_dataset.getImgIds(catIds=list(categories_dict.keys()))
        random.shuffle(image_ids)
        for image_id in image_ids:
            image = coco_dataset.loadImgs([image_id])[0]
            image_path = os.path.join(images_output_dir, image['file_name'])
            if os.path.isfile(image_path):
                continue

            image_data = requests.get(image['coco_url']).content
            with open(image_path, 'wb') as handler:
                handler.write(image_data)

            annotations = coco_dataset.loadAnns(coco_dataset.getAnnIds(imgIds=[image_id], catIds=list(categories_dict.keys())))

            bboxs = []
            cats = []
            for a in annotations:
                if a['category_id'] not in categories_dict:
                    continue

                bboxs.append([a['bbox'][0] / image['width'], a['bbox'][1] / image['height'],
                              a['bbox'][2] / image['width'], a['bbox'][3] / image['height']])
                cats.append(categories_dict[a['category_id']])

            return {'bbox': bboxs, 'category': cats, 'file_path': image_path}

        n_categories -= 1
        if n_categories == 0:
            break

    return None


parser = argparse.ArgumentParser(description='Run calculation of FID score')
parser.add_argument('--unrel_dir', '-u', type=existing_dir, help='Path to UnRel dataset directory',
                    default='../data/datasets/unrel-dataset')
parser.add_argument('--output_dir', '-o', type=new_dir, help='Path to the output dataset',
                    default='../data/datasets/unrel_processed')
parser.add_argument('--coco_metadata', '-c', type=existing_file, help='Path to MSCOCO metadata file',
                    default='../data/datasets/mscoco/instances_val2017.json')

args = parser.parse_args()
unrel_dir = args.unrel_dir
output_dir = args.output_dir
coco = COCO(args.coco_metadata)

out_images_dir = os.path.join(output_dir, 'images')
out_metadata_dir = os.path.join(output_dir, 'metadata')
for dir_name in [out_images_dir, out_metadata_dir]:
    new_dir(dir_name)

unrel_images_dir = os.path.join(unrel_dir, 'images')
unrel_annotations_path = os.path.join(unrel_dir, 'annotations.mat')
unrel_annotations = load_annotations(unrel_annotations_path, unrel_images_dir, out_images_dir)
unrel_to_ms_coco_categories = load_json('../resources/unrel/unrel_mscoco_categories.json')
not_found_categories = set()

for i, ann in tqdm(enumerate(unrel_annotations)):
    unrel_meta_path = os.path.join(out_metadata_dir, f'{str(i).zfill(5)}_a.json')
    if os.path.isfile(unrel_meta_path):
        continue

    unrel_meta = {
        'image_path': ann['file_path'],
        'annotations': {'bbox': ann['bbox'], 'category': ann['category']},
        'category': 'implausible'
    }
    save_json(unrel_meta_path, unrel_meta)

    random.shuffle(ann['category'])
    coco_annotation = find_mscoco_candidates(ann['category'], coco, out_images_dir)
    if coco_annotation is not None:
        coco_meta_path = os.path.join(out_metadata_dir, f'{str(i).zfill(5)}_b.json')
        coco_meta = {
            'image_path': coco_annotation['file_path'],
            'annotations': {'bbox': coco_annotation['bbox'], 'category': coco_annotation['category']},
            'category': 'plausible'
        }
        save_json(coco_meta_path, coco_meta)

print(f'Finished generating dataset. Not found categories: {not_found_categories}')