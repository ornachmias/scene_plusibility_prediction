import argparse
import sys
import os

# Add current path to env (for blender usage)
sys.path.append(os.getcwd())

from utils.file_utils import load_json
from utils.log_utils import get_logger

parser = argparse.ArgumentParser(description='Visualize information')
parser.add_argument('--type', '-t',
                    choices=['scenes_3d_classes', 'scenes_3d_dist', 'render_bbox'],
                    help='Type of visualization')
parser.add_argument('--config', default='./config/visualization.json',
                    help='Path to the visualization configuration file')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

config = load_json(args.config)
logger = get_logger('visualize')

visualize_type = args.type.lower()
if visualize_type == 'scenes_3d_classes':
    from data_generation.scenes_3d_dataset import Scenes3DDataset
    from data_generation.blender.blender_api import BlenderApi
    from visualization.scenes_3d_visualization import Scenes3DVisualization

    scenes_3d = Scenes3DDataset(logger, config)
    scenes_3d.initialize()
    blender_api = BlenderApi(logger)
    scenes_3d_visualize = Scenes3DVisualization(scenes_3d, config['output_dir'])
    scenes_3d_visualize.draw_classes(blender_api)
elif visualize_type == 'render_bbox':
    from visualization.pic_visualization import PicVisualization

    pic_visualization = PicVisualization(config['pic']['root_dir'])
    out_dir = os.path.join(config['output_dir'], 'pic_bboxes')
    pic_visualization.draw_bounding_boxes(out_dir)
elif visualize_type == 'scenes_3d_dist':
    from data_generation.scenes_3d_dataset import Scenes3DDataset
    from visualization.scenes_3d_visualization import Scenes3DVisualization
    scenes_3d = Scenes3DDataset(logger, config)
    scenes_3d.initialize()
    scenes_3d_visualize = Scenes3DVisualization(scenes_3d, config['output_dir'])
    scenes_3d_visualize.draw_categories_distribution()



