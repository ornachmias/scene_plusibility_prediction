import os.path
import random
import time
import multiprocessing
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.camera_generator import CameraGenerator
from data_generation.render_engine import RenderEngine
from data_generation.scenes_3d_dataset import Scenes3DDataset
from data_generation.trasnformations.cooccurrence_location import CooccurrenceLocation
from data_generation.trasnformations.cooccurrence_rotation import CooccurrenceRotation
from data_generation.trasnformations.gravity import Gravity
from data_generation.trasnformations.intersection import Intersection
from data_generation.trasnformations.pose import Pose
from data_generation.trasnformations.size import Size
from data_generation.trasnformations.transformation import Transformation
from utils import log_utils
from utils.file_utils import save_json, load_json


class ImplausibilityEngine:
    def __init__(self, logger: Logger, config: dict, scenes_3d: Scenes3DDataset):
        self.logger = logger
        self.config = config
        self.blender_api = BlenderApi(logger)
        self.scenes_3d = scenes_3d
        self.transform_types = self.config['transformation']['types']
        self.transformations = {}
        self.camera_generator = CameraGenerator(self.logger, self.config, self.scenes_3d)
        self.render = RenderEngine(self.logger, self.config, self.blender_api)
        self.categories_implausibility = load_json(os.path.join(self.config['transformation']['resources'],
                                                                'categories_implausibility.json'))
        self.implausibility_dist = load_json(os.path.join(
            self.config['transformation']['resources'], 'implausibility_distribution.json'))

        self.cache_dir = self.config['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_transformation(self, name) -> Transformation:
        formatted_name = name.lower()
        if formatted_name not in self.transformations:
            if formatted_name == Gravity.name:
                self.transformations[formatted_name] = Gravity(self.logger, self.config,
                                                               self.blender_api, self.scenes_3d)
            elif formatted_name == Size.name:
                self.transformations[formatted_name] = Size(self.logger, self.config,
                                                            self.blender_api, self.scenes_3d)
            elif formatted_name == CooccurrenceLocation.name:
                self.transformations[formatted_name] = CooccurrenceLocation(self.logger, self.config,
                                                                            self.blender_api, self.scenes_3d)
            elif formatted_name == CooccurrenceRotation.name:
                self.transformations[formatted_name] = CooccurrenceRotation(self.logger, self.config,
                                                                            self.blender_api, self.scenes_3d)
            elif formatted_name == Intersection.name:
                self.transformations[formatted_name] = Intersection(self.logger, self.config,
                                                                    self.blender_api, self.scenes_3d)
            elif formatted_name == Pose.name:
                self.transformations[formatted_name] = Pose(self.logger, self.config,
                                                            self.blender_api, self.scenes_3d)
        return self.transformations[formatted_name]

    def get_current_transformation(self, n_transformations):
        current_transform = self.transform_types[n_transformations % len(self.transform_types)]
        current_transform = self.get_transformation(current_transform)
        return current_transform

    def load_objects(self, camera_name):
        objects_in_view = self.blender_api.models_in_camera_view(camera_name)
        objects_in_view = [x for x in objects_in_view if 'room' not in x.lower()]
        self.logger.debug(f'Found {len(objects_in_view)} objects in camera {camera_name} view')
        random.shuffle(objects_in_view)
        return objects_in_view

    def transform(self):
        completed_scenes = {}
        metadata_paths = self.get_metadata_paths()
        is_completed = len(metadata_paths) == len(completed_scenes) and all(list(completed_scenes.values()))
        while not is_completed:
            for metadata_path in metadata_paths:
                completed_scenes[metadata_path] = False
                scene_start_time = time.time()
                try:
                    n_camera_transformations = self.config['transformation']['n_camera_transformations']
                    metadata = load_json(metadata_path)
                    self.logger.info(f'Starting transformation in scene {metadata["scene_id"]}')

                    if 'transformations' not in metadata:
                        metadata['transformations'] = {}

                    n_transformations = len(metadata['transformations'])
                    while n_transformations < self.config['transformation']['n_transformations']:
                        current_transformation = self.get_current_transformation(n_transformations)
                        start_time = time.time()
                        camera_metadata, camera_transformations = self.find_transformation(metadata, current_transformation,
                                                                                           n_camera_transformations)

                        end_time = time.time()
                        found_transformation = True
                        if None in [camera_metadata, camera_transformations] or not camera_transformations:
                            found_transformation = False

                        log_utils.log_performance({
                            'elapsed': end_time - start_time,
                            'transformation': current_transformation.name,
                            'found': found_transformation,
                            'method': 'find_transformation'
                        })

                        if found_transformation:
                            scene_start_time = time.time()
                            self.update_results(metadata_path, camera_metadata, camera_transformations)

                        metadata = load_json(metadata_path)
                        if 'transformations' not in metadata:
                            metadata['transformations'] = {}

                        n_transformations = len(metadata['transformations'])

                        elapsed_time = time.time() - scene_start_time
                        if elapsed_time > 600:
                            raise Exception(f'600 seconds without transformation in scene {metadata["scene_id"]}')

                    self.logger.info(f'Scene {metadata["scene_id"]} completed.')
                    completed_scenes[metadata_path] = True
                except:
                    self.logger.exception('Failed to transform due to an error')

    def update_results(self, metadata_path, camera_metadata, transformations):
        if None in [camera_metadata, transformations]:
            return

        metadata = load_json(metadata_path)
        self.blender_api.load_blend(metadata['blend_path'])
        self.blender_api.create_camera(camera_metadata['location'], camera_metadata['target'], camera_metadata['name'])
        self.blender_api.save_blend(metadata['blend_path'])
        if 'transformations' not in metadata:
            metadata['transformations'] = {}

        metadata['transformations'][camera_metadata['name']] = transformations
        metadata['cameras']['generated'].append(camera_metadata)
        render_name = self.render.render_name(metadata['scene_id'], camera_metadata['name'], 0)
        self.render.render(metadata, camera_metadata['name'], render_name, None)
        categories_count = {}
        for i in range(len(transformations)):
            render_name = self.render.render_name(metadata['scene_id'], camera_metadata['name'], i+1)
            self.get_transformation(transformations[i]['transformation_type']).execute(transformations[i],
                                                                                       transformations[:i])
            self.render.render(metadata, camera_metadata['name'], render_name, transformations[:i+1])
            obj_category = self.scenes_3d.get_category(transformations[i]['obj_name'])[0]
            if obj_category not in categories_count:
                categories_count[obj_category] = 0

            categories_count[obj_category] += 1

        self.update_transformation_count(categories_count)
        self.log_implausibility_dist()
        save_json(metadata_path, metadata)

    def is_transformation_allowed(self, obj_name, transformation_name):
        category, _ = self.scenes_3d.get_category(obj_name)
        allowed_transformations = self.categories_implausibility[category]['allowed']
        if 'co-occurrence_location_rotation' in allowed_transformations:
            allowed_transformations.append('co-occurrence_location')
            allowed_transformations.append('co-occurrence_rotation')

        return transformation_name in allowed_transformations

    def select_transformed_objects(self, obj_names, transformation_name, current_transformations):
        filtered_objs = [x for x in obj_names if self.is_transformation_allowed(x, transformation_name)]
        filtered_objs = [x for x in filtered_objs if x not in [y['obj_name'] for y in current_transformations]]
        if not filtered_objs:
            return None, None

        transformation_count = self.get_transformation_count()
        weights = []

        for obj_name in filtered_objs:
            category = self.scenes_3d.get_category(obj_name)[0]
            weight = self.obj_transformation_weight(transformation_count[category],
                                                    sum(list(transformation_count.values())),
                                                    self.implausibility_dist[category])
            weights.append(weight)

        if len([x for x in weights if x >= 0.00001]) < self.config['transformation']['n_camera_transformations']:
            return filtered_objs, [self.implausibility_dist[self.scenes_3d.get_category(x)[0]] for x in filtered_objs]

        return filtered_objs, weights

    def find_transformation(self, metadata, transformation, n_camera_transformations):
        self.blender_api.load_blend(metadata['blend_path'])
        self.blender_api.use_gpu(self.config['render_resolution'])
        camera_transformations = []

        obj_names = [x for x in self.blender_api.models() if 'room' not in x.lower()]
        camera_metadata = None
        start_time = time.time()

        objs, weights = self.select_transformed_objects(obj_names, transformation.name, camera_transformations)
        self.logger.info(f'Looking transformation for scene {metadata["scene_id"]} out of {len(objs)} objects')
        selected_obj_names = random.choices(objs, weights=weights, k=len(objs) * 10)
        seen = set()
        selected_obj_names = [x for x in selected_obj_names if x not in seen and not seen.add(x)]
        self.logger.debug(f'Objects choices: obj={objs}, weights={weights}, selected_obj_names={selected_obj_names}')
        for obj_name in selected_obj_names:
            if obj_name in [x['obj_name'] for x in camera_transformations]:
                continue

            if time.time() - start_time > self.config['idle_timeout']:
                self.logger.warning('Reached idle timeout, using currently found data')
                return camera_metadata, camera_transformations

            for retry_count in range(self.config['transformation']['object_retries']):
                manager = multiprocessing.Manager()
                result_dict = manager.dict()
                performance_start_time = time.time()
                p = multiprocessing.Process(target=self.find_single_transformation,
                                            name='find_single_transformation',
                                            args=(obj_name, metadata, transformation,
                                                  camera_transformations, result_dict,))
                p.start()
                p.join(self.config['transformation_timeout'])
                if p.is_alive():
                    self.logger.warning(f'Transformation took too long to complete, killing process.')
                    p.kill()
                    p.join()

                log_utils.log_performance({
                    'elapsed': time.time() - performance_start_time,
                    'transformation': transformation.name,
                    'found': 'is_found' in result_dict and result_dict['is_found'],
                    'completed': 'is_found' in result_dict,
                    'method': 'find_single_transformation',
                    'current_transformation_idx': len(camera_transformations) + 1
                })

                if 'is_found' in result_dict and result_dict['is_found']:
                    camera_transformations.append(result_dict['transformation_params'])
                    transformation.execute(camera_transformations[-1], camera_transformations[:-1])
                    camera_metadata = result_dict['camera_metadata']
                    start_time += self.config['idle_timeout']
                    break

            self.logger.info(f'Current camera has {len(camera_transformations)} transformations')
            if len(camera_transformations) >= n_camera_transformations:
                break

        if not camera_transformations or len(camera_transformations) < 4:
            return None, None

        return camera_metadata, camera_transformations

    def find_single_transformation(self, obj_name, metadata, transformation,
                                   current_camera_transformations, result_dict):
        result_dict['is_found'] = False
        result_dict['transformation_params'] = None
        result_dict['camera_metadata'] = None

        start_time = time.time()
        transformation_params = transformation.find(obj_name, metadata)
        log_utils.log_performance({
            'elapsed': time.time() - start_time,
            'transformation': transformation.name,
            'found': transformation_params is not None and 'transform' in transformation_params,
            'method': 'find'
        })

        if not transformation_params or not transformation_params['transform']:
            return

        if any([x < 0 for x in transformation_params['transform']['location']]):
            return

        current_transformations = [x for x in current_camera_transformations]
        current_transformations.append(transformation_params)
        self.logger.info(f'Looking for valid camera for: {", ".join([x["obj_name"] for x in current_transformations])}')
        start_time = time.time()
        camera_metadata = self.camera_generator.find_camera_by_transformation(metadata, current_transformations)
        log_utils.log_performance({
            'elapsed': time.time() - start_time,
            'transformation': transformation.name,
            'found': camera_metadata is not None,
            'completed': 'is_found' in result_dict,
            'method': 'find_camera_by_transformation',
            'current_transformation_idx': len(current_transformations)
        })

        if camera_metadata is None:
            transformation.revert(transformation_params)
            self.logger.info('No camera was found')
            return
        self.logger.info('Valid camera found')

        start_time = time.time()
        is_valid = transformation.validate(camera_metadata['name'], metadata, transformation_params, current_camera_transformations)
        log_utils.log_performance({
            'elapsed': time.time() - start_time,
            'transformation': transformation.name,
            'found': is_valid,
            'method': 'validate'
        })

        if not is_valid:
            self.logger.debug('Transformation marked as invalid')
            transformation.revert(transformation_params)
            return
        else:
            self.logger.info(f'Found model transformation. Object: {obj_name}')
            result_dict['is_found'] = True
            result_dict['transformation_params'] = transformation_params
            result_dict['camera_metadata'] = camera_metadata
            return

    def revert_transformations(self):
        for metadata_path in self.get_metadata_paths():
            metadata = load_json(metadata_path)
            self.blender_api.load_blend(metadata['blend_path'])
            for camera in self.blender_api.cameras():
                if 'origin' not in camera:
                    self.blender_api.delete(camera)

            metadata['cameras']['generated'] = []
            metadata['transformations'] = {}
            save_json(metadata_path, metadata)
            self.blender_api.save_blend(metadata['blend_path'])

    def get_metadata_paths(self):
        metadata_dir = self.config['metadata_dir']
        if self.config['debug_scene'] is not None:
            return [os.path.join(metadata_dir, self.config['debug_scene'] + '.json')]

        return [os.path.join(metadata_dir, x) for x in os.listdir(metadata_dir) if x.endswith('.json')]

    def get_transformation_count(self):
        path, current_count = self.create_transformation_count()
        return current_count

    def update_transformation_count(self, value):
        path, current_count = self.create_transformation_count()
        for category in value:
            current_count[category] += value[category]

        path = os.path.join(self.cache_dir, 'transformation_count.json')
        save_json(path, current_count)

    def create_transformation_count(self):
        path = os.path.join(self.cache_dir, 'transformation_count.json')
        if os.path.isfile(path):
            return path, load_json(path)

        current_count = {x: 0 for x in self.implausibility_dist}
        save_json(path, current_count)
        return path, current_count

    @staticmethod
    def obj_transformation_weight(transformed, total_transformed, implausibility_dist):
        if total_transformed == 0:
            return implausibility_dist

        return max(0, implausibility_dist - (transformed / total_transformed))

    def log_implausibility_dist(self):
        transformation_count = self.get_transformation_count()
        transformations_sum = sum(list(transformation_count.values()))
        if transformations_sum == 0:
            return

        transformation_dist = {x: transformation_count[x]/transformations_sum for x in transformation_count}
        distance = 0
        for category in self.implausibility_dist:
            dest_dist = self.implausibility_dist[category]
            current_dist = transformation_dist[category]
            distance += abs(dest_dist - current_dist)

        self.logger.info(f'Current transformation distribution distance: {distance}. Details: {transformation_dist}')

