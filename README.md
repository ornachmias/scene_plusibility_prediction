# Plausibility in Context
## Data Generation
From the generate.py code directory, enter the following command:
### Windows
```
<blender_dir>\blender.exe --background --python generate.py --
```

### Ubuntu/MacOS
```
<blender_dir>/blender --background --python generate.py --
```

The script is self-contained, and it will download all necessary files, process them and output the matching renders and metadata.
You may modify the relevant configuration file in `config/data_generation.json`.

In case of a problem in the process, we suggest modifying the `debug_scene` configuration value to a name of on of the scenes (e.g. `"debug_scene": "sharonMichaelLiving"`), 
that way the process will work end-to-end based on a single scene, and will allow you to detect problem faster.

## Training and Evaluation
In order to train all relevant model, use the following script:

```
python3 train.py --config <experiment_config_file>
```

`<experiment_config_file>` should be replaced by one of the following:
- `./experiments/binary_classification.json`
- `./experiments/classification.json`
- `./experiments/regression.json`

This script will run both training and evaluation process.

## Other Experiments
In addition to the original data creation, training and evaluation script, there are some additional script that can be run
to reproduce the entire results published in our paper:
- `scripts/build_unrel_dataset.py` - Will build the UnRel dataset we used, with a mix of MSCOCO images. Please note that UnRel dataset and MSCOCO metadata file should be downloaded manually, and may require to configure the `unrel_dir` and `coco_metadata` based on the downloaded paths.
- `scripts/count_images_by_scene.py` - A debug/progress script to count the number of plausible and implausible images for each scene.
- `scripts/evaluate_bc.py` - Binary classification evaluation and plots generation.
- `scripts/evaluate_mcc.py` - Multi class classification evaluation and plots generation.
- `scripts/evaluate_reg.py` - Regression evaluation and plots generation.
- `scripts/evaluate_unrel.py` - UnRel dataset evaluation and plots generation.
- `scripts/fid_evaluation_bc.py` - FID score calculation for plausible/implausible images and plots generation.
- `scripts/fid_evaluation_category.py` - FID score calculation for plausible/implausible tpes images and plots generation.

## Human Evaluation
The human evaluation was made using Amazon's Mechanical Turk, that access a remote host that contains the images for classification.
In order to create a HIT in AMT, please perform the following steps:
1. On the host that contains the images, enable external web traffic.
2. Select example images and edit `human_evaluation/build_csv.py` file according to those images.
3. On the host, run the following commands: `nohup gunicorn --bind 0.0.0.0:5000  --limit-request-line 0 --workers 4 --log-file ./logs/gunicorn.log wsgi:app > ./logs/wsgl.out &`. It'll start and HTTP service that will enable to access the images from AMT.
4. On the host, run the following script: `human_evaluation/build_csv`, please make sure to initialize the required parameters based on the base URL for your host.
5. Use the CSV output to initialize the questions in AMT.

You can review the results using `human_evaluation/review_results.py`.
