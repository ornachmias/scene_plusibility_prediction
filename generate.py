import argparse
import os
import sys

# Add current path to env (for blender usage)
sys.path.append(os.getcwd())

from utils.log_utils import get_logger, hide_pil_logs
from utils.file_utils import load_json
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.builder import Builder
from data_generation.implausibility_engine import ImplausibilityEngine

logger = get_logger('generate')
hide_pil_logs()

parser = argparse.ArgumentParser(description='Generate the full PIC dataset')
parser.add_argument('--config', default='./config/data_generation.json',
                    help='Path to the generation configuration file')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)
config = load_json(args.config)
logger.info(f'Input parameters:\n{config}')

scenes_3d = Scenes3DDataset(logger, config)
scenes_3d.initialize()

if config['debug_scene'] is not None:
    scenes_3d.set_debug()

for step in config['steps']:
    if step == 'build':
        scene_builder = Builder(logger, config, scenes_3d)
        scene_builder.build_scenes()
    elif step == 'undo_transform':
        engine = ImplausibilityEngine(logger, config, scenes_3d)
        engine.revert_transformations()
    elif step == 'transform':
        engine = ImplausibilityEngine(logger, config, scenes_3d)
        engine.transform()
    else:
        logger.warning('Unrecognized action, exiting generation script.')
        exit(1)
