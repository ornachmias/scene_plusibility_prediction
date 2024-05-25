import random
import time
from logging import Logger

from data_generation.blender.blender_api import BlenderApi
from data_generation.scenes_3d_dataset import Scenes3DDataset
from utils import log_utils
from utils.math_utils import random_points_above


class CameraGenerator:
    def __init__(self, logger: Logger, config: dict, scenes_3d: Scenes3DDataset):
        self.logger = logger
        self.scenes_3d = scenes_3d
        self.config = config
        self.metadata_dir = self.config['metadata_dir']
        self.blender_api = BlenderApi(logger)

    def find_camera_by_transformation(self, metadata, transformations_metadata):
        generated_cameras = metadata['cameras']['generated']
        models_locations = [metadata['models'][x]['location'] for x in metadata['models'] if 'room' not in x.lower()]
        mean_location = [0, 0, 0]

        for i in range(3):
            current_avg = 0
            for t in transformations_metadata:
                current_avg += t['original']['location'][i]
                current_avg += t['transform']['location'][i]

            current_avg = current_avg / (len(transformations_metadata) * 2)
            mean_location[i] = current_avg

        suggested_locations = self.suggest_locations(mean_location)
        for suggested_location in suggested_locations:
            name = self.generate_camera_name(len(generated_cameras))
            if name in self.blender_api.cameras():
                self.blender_api.delete(name)

            name, location, target = self.create_camera(suggested_location, models_locations, name)
            self.logger.debug(f'[scene_id:{metadata["scene_id"]}] Testing suggested location: {suggested_location}, target: {target}')
            start_time = time.time()
            location, target = self.locate_camera(transformations_metadata, name)
            log_utils.log_performance({
                'elapsed': time.time() - start_time,
                'transformation': transformations_metadata[0]['transformation_type'],
                'found': location is not None,
                'completed': location is not None,
                'method': 'locate_camera',
                'current_transformation_idx': len(transformations_metadata)
            })

            if None in [location, target]:
                self.blender_api.delete(name)
                continue

            self.blender_api.set_location(name, location)
            self.blender_api.adjust_camera(name, target)
            self.logger.debug(f'Suggested camera {name} was marked as valid. '
                              f'Location: {location[:]}, Target: {target[:]}')

            return {'name': name, 'location': location[:], 'target': target[:]}

        return None

    def is_camera_valid(self, name):
        resolution = self.config['render_resolution']
        camera_configs = self.config['camera_generation']
        absolute_models_threshold = camera_configs['absolute_models_threshold']
        relative_models_threshold = camera_configs['relative_models_threshold']

        self.blender_api.set_resolution(resolution)
        all_models = self.blender_api.models()
        visible_models = self.blender_api.models_in_camera_view(name)

        is_valid_absolute = len(visible_models) >= absolute_models_threshold
        if len(all_models) > 30:
            is_valid_relative = len(visible_models) > 10
        else:
            is_valid_relative = (len(visible_models) / len(all_models)) >= relative_models_threshold
        is_valid = is_valid_relative and is_valid_absolute
        return is_valid

    @staticmethod
    def generate_camera_name(num_existing_cameras, height=None):
        if height is None:
            return 'generated_camera_{}'.format(str(num_existing_cameras).zfill(3))
        else:
            return 'generated_camera_{}_{}'.format(str(num_existing_cameras).zfill(3), str(height).zfill(3))

    def random_points_above(self, initial_location, radius, n_locations):
        return random_points_above(initial_location, radius, n_locations, (self.config['camera_generation']['min_z'],
                                                                           self.config['camera_generation']['max_z']))

    def clamp_to_room(self, location):
        room_name = None
        for model in self.blender_api.models():
            if 'room' in model:
                room_name = model
                break

        if room_name is None:
            return location

        room_bbox_min = self.blender_api.get_3d_bbox_min(room_name)[:]
        room_bbox_max = self.blender_api.get_3d_bbox_max(room_name)[:]
        new_location = [x for x in location]
        for i in range(3):
            new_location[i] = max(room_bbox_min[i], min(room_bbox_max[i], location[i]))

        return new_location

    def suggest_locations(self, initial_location):
        configs = self.config['camera_generation']['location_suggestion']
        radius = configs['radius']
        n_suggested_locations = configs['n_locations']
        suggested_locations = []

        while len(suggested_locations) < n_suggested_locations:
            current_suggested_locations = self.random_points_above(initial_location, radius, n_suggested_locations)
            current_suggested_locations = [self.clamp_to_room(t_loc) for t_loc in current_suggested_locations]
            if len(current_suggested_locations) > 0:
                suggested_locations.extend(current_suggested_locations)

        return suggested_locations[:n_suggested_locations]

    def create_camera(self, location, targets, name):
        if name in self.blender_api.cameras():
            self.blender_api.delete(name)

        random.shuffle(targets)
        selected_target = targets[0]
        self.blender_api.create_camera(location, selected_target, name)
        self.blender_api.clamp_to_room(name)
        self.blender_api.set_resolution(self.config['render_resolution'])
        return name, location, selected_target

    def generate_duplicate(self, obj_name, index, transformation):
        duplicate_name = self.blender_api.duplicate(obj_name, index)
        self.blender_api.set_location(duplicate_name, transformation['location'])
        self.blender_api.set_rotation(duplicate_name, transformation['rotation'])
        self.blender_api.set_scale(duplicate_name, transformation['scale'])
        return duplicate_name

    def locate_camera(self, transformations_metadata, camera_name):
        adjuster_name = 'adjuster'
        transformation_names = set()
        duplicate_names = []
        duplicate_relations = {}
        try:
            for transformation_metadata in transformations_metadata:
                obj_name = transformation_metadata['obj_name']
                transformation_names.add(transformation_metadata['transformation_type'])
                duplicate_relations[obj_name] = {
                    'original': self.generate_duplicate(obj_name, 1, transformation_metadata['original']),
                    'transform': self.generate_duplicate(obj_name, 2, transformation_metadata['transform'])
                }
                duplicate_names.extend([duplicate_relations[obj_name]['original'],
                                        duplicate_relations[obj_name]['transform']])

            mean_location = [0, 0, 0]
            for i in range(len(mean_location)):
                mean_location[i] = sum([self.blender_api.location(x)[i] for x in duplicate_names]) / len(duplicate_names)

            self.blender_api.create_empty_sphere(adjuster_name, mean_location)
            self.blender_api.point_camera_at(camera_name, self.blender_api.location(adjuster_name))

            move_vector = self.blender_api.build_movement_vector(self.blender_api.location(camera_name),
                                                                 self.blender_api.location(adjuster_name))

            # Search - Step 1
            step_size = 10
            max_iterations = 20
            directions_params = [
                {'iteration': 0, 'current_location': self.blender_api.location(camera_name),
                 'last_location': self.blender_api.location(camera_name), 'step_size': step_size,
                 'occlusions': False},
                {'iteration': 0, 'current_location': self.blender_api.location(camera_name),
                 'last_location': self.blender_api.location(camera_name), 'step_size': step_size * -1,
                 'occlusions': False}
            ]

            current_iteration = 0
            while all([x['iteration'] < max_iterations for x in directions_params]):
                selected_direction = current_iteration % 2

                pos = directions_params[selected_direction]['step_size']
                new_location = [directions_params[selected_direction]['last_location'][i] +
                                ((int(current_iteration / 2) + 1) * pos * move_vector[i])
                                for i in range(3)]

                self.blender_api.set_location(camera_name, new_location)
                directions_params[selected_direction]['current_location'] = self.blender_api.location(camera_name)

                target = self.blender_api.location(adjuster_name)
                self.blender_api.point_camera_at(camera_name, target)
                self.blender_api.prepare_render(camera_name, self.config['render_resolution'])
                self.blender_api.update()

                bbox_visibility = self.test_bbox_visibility(camera_name, duplicate_relations)
                no_occlusions = self.test_occlusions(camera_name, duplicate_relations)
                min_objects = self.is_camera_valid(camera_name)
                self.logger.debug(f'Testing found that for transformations {list(transformation_names)}: '
                                  f'bbox_visibility={bbox_visibility}, '
                                  f'no_occlusions={no_occlusions}, '
                                  f'visible_objects={min_objects} ('
                                  f'{len(self.blender_api.models_in_camera_view(camera_name))})')
                if not no_occlusions:
                    if directions_params[selected_direction]['occlusions']:
                        break
                    else:
                        directions_params[selected_direction]['occlusions'] = True

                if bbox_visibility and no_occlusions and min_objects:
                    return new_location, target
                else:
                    directions_params[selected_direction]['iteration'] += 1
                    current_iteration += 1

            return None, None
        finally:
            self.blender_api.delete_empties()
            for duplicate in duplicate_names:
                self.blender_api.delete(duplicate)

    def bbox_visibility(self, model_name, camera_name, resolution):
        self.blender_api.set_resolution(resolution)
        raw_bbox = list(self.blender_api.bounding_box_2d(camera_name, model_name, resolution))
        raw_area = abs((raw_bbox[3] - raw_bbox[1]) * (raw_bbox[2] - raw_bbox[0]))
        refined_bbox = list(self.blender_api.bounding_box_2d(camera_name, model_name, resolution, True))
        refined_area = abs((refined_bbox[3] - refined_bbox[1]) * (refined_bbox[2] - refined_bbox[0]))
        if raw_area == 0:
            return 0

        return refined_area / raw_area

    def test_bbox_visibility(self, camera_name, duplicate_relations):
        resolution = self.config['render_resolution']

        for obj_name in duplicate_relations:
            for duplicate_type in ['original', 'transform']:
                bbox = self.blender_api.bounding_box_2d(camera_name, duplicate_relations[obj_name][duplicate_type], resolution, True)
                if abs(bbox[0] - bbox[2]) == resolution and abs(bbox[1] - bbox[3]) == resolution:
                    return False

                if not self.blender_api.is_valid_bbox(bbox, resolution):
                    return False

                bbox_visibility = self.bbox_visibility(duplicate_relations[obj_name][duplicate_type],
                                                       camera_name, resolution)
                if bbox_visibility < 0.5:
                    return False

        return True

    def test_occlusions(self, camera_name, duplicate_relations):
        for obj_name in duplicate_relations:
            transform_obj = duplicate_relations[obj_name]['transform']
            original_obj = duplicate_relations[obj_name]['original']

            self.blender_api.set_hide(original_obj, False)
            self.blender_api.set_hide(transform_obj, True)
            if self.blender_api.is_occluded(camera_name, original_obj):
                self.blender_api.set_hide(transform_obj, False)
                return False

            self.blender_api.set_hide(transform_obj, False)
            self.blender_api.set_hide(original_obj, True)
            if self.blender_api.is_occluded(camera_name, transform_obj):
                self.blender_api.set_hide(original_obj, False)
                return False

            self.blender_api.set_hide(original_obj, False)

        return True
