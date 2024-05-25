import os
import random

from flask import Flask, render_template, send_from_directory, request

from utils.file_utils import load_json, save_json

template_dir = os.path.abspath('./data_service/templates')
static_dir = os.path.abspath('./data_service/static')
renders_dir = './data/datasets/pic/render'
examples_dir = './data/datasets/pic/examples'


def get_images_to_validate():
    implausible_dir = os.path.join(renders_dir, 'implausible')
    result = {}
    skip_keys = set()
    for category in os.listdir(implausible_dir):
        category_dir = os.path.join(implausible_dir, category)
        for filename in os.listdir(category_dir):
            base_name = ".".join(filename.split('.')[:-1])
            if base_name not in result:
                result[base_name] = {}

            path = os.path.join(category_dir, filename)
            if filename.endswith('.png'):
                result[base_name]['implausible_image_path'] = os.path.join('implausible', category, filename)
            elif filename.endswith('.json'):
                metadata = load_json(path)
                if 'valid_result' in metadata and metadata['valid_result'] in [True, False]:
                    skip_keys.add(base_name)
                    continue
                else:
                    result[base_name]['metadata_path'] = path
            else:
                continue

            # <scene_id>.<camera_name>.<n_transformed>.png
            if 'plausible_image_path' not in result[base_name]:
                plausible_image_name = ".".join(base_name.split('.')[:-1]) + '.0000.png'
                plausible_image_path = os.path.join('plausible', plausible_image_name)
                result[base_name]['plausible_image_path'] = plausible_image_path

    for skip_key in skip_keys:
        result.pop(skip_key)

    return result


validate_list = get_images_to_validate()
app = Flask(__name__, static_url_path='', template_folder=template_dir, static_folder=static_dir)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/view')
def view():
    categories = os.listdir(os.path.join(renders_dir, 'implausible'))
    return render_template('view.html', categories=categories)


@app.route('/validate', methods=('GET', 'POST'))
def validate():
    if request.method == 'POST':
        base_name = request.form['base_name']
        metadata_path = validate_list[base_name]['metadata_path']
        metadata = load_json(metadata_path)

        if request.form['submit_button'] == 'Skip':
            pass  # Don't do anything, just pop it from dict
        elif request.form['submit_button'] == 'Valid':
            metadata['valid_result'] = True
            save_json(metadata_path, metadata)
        elif request.form['submit_button'] == 'Invalid':
            metadata['valid_result'] = False
            save_json(metadata_path, metadata)
        else:
            pass

        validate_list.pop(base_name)

    k = random.choice(list(validate_list.keys()))
    v = validate_list[k]
    return render_template('validate.html', implausible_image_path=v['implausible_image_path'],
                           plausible_image_path=v['plausible_image_path'],
                           n_images_left=len(validate_list),
                           base_name=k)


@app.route('/implausible/<category>')
def get_images_in_category(category):
    selected_dir = os.path.join(renders_dir, 'implausible', category)
    paths = [os.path.join('.', 'implausible', category, x) for x in os.listdir(selected_dir)]
    return render_template('implausibility_list.html', paths=paths, category=category)


@app.route('/resources/<path:path>')
def resources(path):
    return send_from_directory(renders_dir, path)


@app.route('/examples/<path:path>')
def examples(path):
    return send_from_directory(examples_dir, path)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
