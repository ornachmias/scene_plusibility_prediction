import os
import math
import shutil

import matplotlib.pyplot as plt
from PIL import Image

from data_generation.scenes_3d_dataset import Scenes3DDataset


class Scenes3DVisualization:
    def __init__(self, dataset: Scenes3DDataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_classes(self, blender_api):
        combined_dir = os.path.join(self.output_dir, 'categories')
        single_objs_dir = os.path.join(combined_dir, 'singles')
        models = self.dataset.get_models_by_category()
        for category in models:
            category_dir = os.path.join(single_objs_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            for i, model in enumerate(models[category]):
                render_path = os.path.join(category_dir, f'{str(i).zfill(4)}.png')
                blender_api.render_model(model, render_path, 300)

            self.build_figure(category, category_dir, combined_dir, models[category])

        shutil.rmtree(single_objs_dir, ignore_errors=True)

    def draw_categories_distribution(self):
        categories = {}
        for scene_id in self.dataset.scene_ids:
            models = self.dataset.get_models(scene_id)
            for model in models:
                if model['category'] not in categories:
                    categories[model['category']] = 0

                categories[model['category']] += 1

        categories.pop('room')
        categories.pop('uncategorized')

        for category in sorted(categories.keys()):
            print(f'{categories[category]/sum(list(categories.values())):.3f}')

        plt.figure(figsize=(16, 12), dpi=80)
        plt.bar(categories.keys(), categories.values())
        plt.title('Scenes 3D Categories Distribution')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(os.path.join(self.output_dir, 'categories_dist.png'))

    @staticmethod
    def build_figure(category, category_dir, combined_dir, models):
        n_cols = math.sqrt(len(models))
        n_rows = math.sqrt(len(models))
        if n_cols > int(n_cols):
            n_cols += 1

        if n_rows > int(n_rows):
            n_rows += 1

        n_cols = int(n_cols)
        n_rows = int(n_rows)
        fig = plt.figure()
        plt.title(category)
        file_paths = [os.path.join(category_dir, x) for x in os.listdir(category_dir)]
        for i, file_path in enumerate(file_paths):
            img = Image.open(file_path)
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.imshow(img)
            ax.set_axis_off()

        plt.axis('off')
        plt.savefig(os.path.join(combined_dir, f'{category}.png'))











