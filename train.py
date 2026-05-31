import os
# Influences performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from emtrain.models import build_model
from emtrain.utils.comet.comet_log import comet_log_batch
from emtrain.utils.training.prep import prep_training_experiment

import argparse
import comet_ml
import gunpowder as gp
import json
import logging
import numpy as np
import tempfile
import torch

from datetime import datetime
from glob import glob
from lsd.train.gp import AddLocalShapeDescriptor


# TODO: Auto new project creation

logging.basicConfig(level=logging.INFO)


def start_train(project_dir,
                GPU_ID,
                num_workers,
                resume_training,
                training_config=None,
                no_comet_log=False
                ):

    project_dir = os.path.abspath(project_dir)
    project_name = project_dir.split('/')[-1]
    year = datetime.now().year-2000
    exp_name = f'{year}_{project_name}_'
    existing_projects = glob(os.path.join(project_dir,f'*{exp_name}*/'))

    if resume_training is not None and len(existing_projects)>0:
        # Override exp_name
        if resume_training == '-1':
            # Continue with the latest bout
            experiment_dir = os.path.abspath(sorted(existing_projects)[-1])
            exp_name = experiment_dir.split('/')[-1]
            logging.info(f'Resuming latest experiment: {exp_name}')
        else:
            # Continue with provided experiment name
            experiment_dir = os.path.abspath(os.path.join(project_dir, resume_training))
            exp_name = experiment_dir.split('/')[-1]
            logging.info(f'Resuming experiment: {exp_name}')
        training_config = os.path.join(experiment_dir, 'training_config.json')
    else:
        assert training_config is not None, 'Please provide a training configuration to start a new experiment.'

        with open(training_config, 'r') as f:
            experiment_dir = json.load(f).get('experiment_dir')

        if experiment_dir is None:
            existing_projects = glob(os.path.join(project_dir,f'*{exp_name}*/'))
            index = len([p for p in existing_projects if exp_name in p])
            exp_name += str(index).zfill(2)
            experiment_dir = os.path.join(project_dir, exp_name)
            logging.info(f'Starting new experiment: {exp_name}')
        else:
            exp_name = os.path.abspath(experiment_dir).split('/')[-1]
            logging.info(f'Starting existing experiment: {exp_name}')

    logging.info('Experiment dir:')
    logging.info(f'    {experiment_dir}')

    # Get configs
    training_config, ground_truth_config, model_config, augment_config = prep_training_experiment(experiment_dir=experiment_dir,
                                                                                                  training_config=training_config,
                                                                                                  GPU_ID=GPU_ID,
                                                                                                  num_workers=num_workers)

    # Training parameters
    num_iterations  = training_config['training']['num_iterations']
    save_every      = training_config['training']['save_every']
    snapshots_every = training_config['training']['snapshots_every']
    cache_size      = training_config['training']['cache_size']

    # Ground-truth config
    gt_datasets = training_config['ground_truth']['datasets']
    ground_truth_data = []
    for dataset_name in gt_datasets:
        dataset = ground_truth_config[dataset_name]

        raw_data = dataset['raw_data']
        ground_truth = dataset['ground_truth']
        gt_zarr_dataset = dataset['ground_truth_dataset']

        for key, ground_truths in ground_truth.items():
            raw_path = raw_data[key]
            for gt_path in ground_truths:
                ground_truth_data.append((raw_path, gt_path, gt_zarr_dataset))

    # Augmentation parameters
    elastic_spacing     = augment_config['elastic_spacing']
    elastic_jitter      = augment_config['elastic_jitter']
    prob_elastic        = augment_config['prob_elastic']
    intensity_scmin     = augment_config['intensity_scmin']
    intensity_scmax     = augment_config['intensity_scmax']
    intensity_shmin     = augment_config['intensity_shmin']
    intensity_shmax     = augment_config['intensity_shmax']
    prob_noise          = augment_config['prob_noise']
    prob_missing        = augment_config['prob_missing']
    prob_low_contrast   = augment_config['prob_low_contrast']

    logging.info('Starting training...')

    if not no_comet_log and not os.path.exists(os.path.join(experiment_dir, '.comet_exp_key')):
        # New experiment
        comet_exp = comet_ml.start(project=project_name,
                                   project_name=project_name)
        comet_exp.set_name(exp_name)
        comet_exp.log_parameters(training_config)
        comet_exp.log_parameters(ground_truth_config)
        comet_exp.log_parameters(augment_config)

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, '.comet_exp_key'), 'w') as f:
            f.write(comet_exp.get_key())
    elif not no_comet_log:
        # Resume an existing experiment
        with open(os.path.join(experiment_dir, '.comet_exp_key'), 'r') as f:
            exp_key = f.read()
        comet_exp = comet_ml.start(experiment_key=exp_key)
    else:
        logging.warning('\x1b[1;31m' + 'Logging to comet is disabled.' + '\x1b[0m')

    train(experiment_dir=experiment_dir,
          GPU_ID=GPU_ID,
          num_workers=num_workers,
          cache_size=cache_size,
          ground_truth_data=ground_truth_data,
          model_config=model_config,
          elastic_spacing=elastic_spacing,
          elastic_jitter=elastic_jitter,
          prob_elastic=prob_elastic,
          intensity_scmin=intensity_scmin,
          intensity_scmax=intensity_scmax,
          intensity_shmin=intensity_shmin,
          intensity_shmax=intensity_shmax,
          prob_noise=prob_noise,
          prob_missing=prob_missing,
          prob_low_contrast=prob_low_contrast,
          num_iterations=num_iterations,
          save_every=save_every,
          snapshots_every=snapshots_every
          )

def train(experiment_dir,
          GPU_ID,
          num_workers, 
          cache_size,
          ground_truth_data,
          num_fmaps,
          input_shape,
          output_shape,
          voxel_size,
          elastic_spacing,
          elastic_jitter,
          prob_elastic,
          intensity_scmin,
          intensity_scmax,
          intensity_shmin,
          intensity_shmax,
          prob_noise,
          prob_missing,
          prob_low_contrast,
          num_iterations,
          save_every,
          snapshots_every
         ):
    
    model_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    checkpoints = [0] + [int(c.rsplit('_', maxsplit=1)[-1]) for c in glob(model_dir + '/*')]
    prev_iter = max(checkpoints)

    temp_dir = os.path.join(experiment_dir, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    tempfile.tempdir = temp_dir
    
    profile_every = 10

    # create model, loss, and optimizer
    # compute a valid input and output size

    # for x and y:
    #
    # 124 -> 120                            40 -> 36 (with stride 8: 32)
    #         |                              |
    #         60 -> 56                24 -> 20
    #                |                 |
    #               28 -> 24    16 -> 12
    #                      |     |
    #                     12 ->  8
    #

    #
    # for z:
    #
    # 36 -> 32                            12 -> 8
    #       |                             |
    #       32 -> 28                16 -> 12
    #             |                 |
    #             28 -> 24    20 -> 16
    #                    |    |
    #                   24 -> 20

    voxel_size = gp.Coordinate(voxel_size)
    input_shape = gp.Coordinate(input_shape)   # shape = voxel size
    output_shape = gp.Coordinate(output_shape)
    input_size = input_shape * voxel_size   # size = world unit
    output_size = output_shape * voxel_size

    # Initiate model
    model = AffsLsdModel(num_fmaps=num_fmaps)
    loss = AffLsdLoss()
    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())

    # Declare gunpowder arrays
    raw = gp.ArrayKey('RAW')
    seg = gp.ArrayKey('SEGMENTATION')
    affs = gp.ArrayKey('AFFINITIES')
    pred_affs = gp.ArrayKey('PRED_AFFINITIES')
    lsds = gp.ArrayKey('LSDS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')

    # Merge samples
    sources = [(gp.ZarrSource(store=gt[1],
                              datasets={seg: gt[2]},
                              array_specs={seg: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)}),
                gp.ZarrSource(store=gt[0],
                              datasets={raw: 'raw'},
                              array_specs={raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)})
               ) +
                gp.MergeProvider() +
                gp.RandomLocation()
              for gt in ground_truth_data]

    # Create pipeline   
    pipeline = tuple(sources) + gp.RandomProvider()  

    # Normalize raw greyscale data
    pipeline += gp.Normalize(raw)

    # Augmentations
    pipeline += gp.DeformAugment(control_point_spacing=gp.Coordinate(elastic_spacing),
                                 jitter_sigma=gp.Coordinate(elastic_jitter),
                                 subsample=8,
                                 p=prob_elastic,
                                 rotate=False)
    pipeline += gp.SimpleAugment(transpose_only=[1, 2]) # transposes in dimensions [0, 1, 2]
    pipeline += gp.IntensityAugment(raw, 
                                    intensity_scmin, 
                                    intensity_scmax, 
                                    intensity_shmin,
                                    intensity_shmax, 
                                    z_section_wise=True)
    pipeline += gp.NoiseAugment(raw, 
                                p=prob_noise)  # change var if noise is too extreme here (variance of std dev)
    pipeline += gp.DefectAugment(raw, 
                                 prob_missing=prob_missing,
                                 prob_low_contrast=prob_low_contrast,
                                 prob_deform=0)
    
    # Create affinities
    pipeline += gp.AddAffinities([[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                 seg,
                                 affs,
                                 dtype='float32')
    pipeline += gp.BalanceLabels(affs, 
                                 affs_weights)

    # Add local shape descriptors
    pipeline += AddLocalShapeDescriptor(seg, 
                                        lsds, 
                                        sigma=60.0)

    # we have:
    # raw:  (d, h, w)
    # affs: (3, d, h, w)
    # lsds: (10, d, h, w)

    # what torch wants:
    # raw:  (b=1, c=1, d, h, w)
    # affs: (b=1, c=3, d, h, w)
    # lsds: (b=1, c=10, d, h, w)

    pipeline += gp.Unsqueeze([raw]) # add a dim to raw

    # we have:
    # raw:  (1, d, h, w)
    # affs: (3, d, h, w)
    # lsds: (10, d, h, w)

    pipeline += gp.Stack(1)

    # we have:
    # raw:  (1, 1, d, h, w)
    # affs: (1, 3, d, h, w)
    # lsds: (1, 10, d, h, w)

    pipeline += gp.PreCache(cache_size=cache_size, 
                            num_workers=num_workers)     # pre-load batchs, increases speed

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_affs,
            1: pred_lsds
        },
        loss_inputs={
            'pred_affs': pred_affs,
            'affs': affs,
            'affs_weights': affs_weights,
            'pred_lsds': pred_lsds,
            'lsds': lsds
        },
        checkpoint_basename=os.path.join(model_dir, 'model'),
        device=f'cuda:{','.join([str(g) for g in GPU_ID])}',
        save_every=save_every)

    pipeline += gp.Squeeze([raw])           # re-squeeze arrays to have n dim fitting snapshot node
    pipeline += gp.Squeeze([raw, 
                            seg,
                            affs, 
                            pred_affs,
                            lsds,
                            pred_lsds
                            ])

    # we have:
    # raw:       (d, h, w)
    # affs:      (3, d, h, w)
    # pred_affs: (3, d, h, w)
    # lsds:      (10, d, h, w)
    # pred_lsds: (10, d, h, w)

    pipeline += gp.Snapshot({
                                raw: 'raw',
                                seg: 'gt_seg',
                                affs: 'affs',
                                pred_affs: 'pred_affs',
                                lsds: 'lsds',
                                pred_lsds: 'pred_lsds',
                            },
                            output_dir=os.path.join(experiment_dir, 'snapshots'),
                            every=snapshots_every) 

    pipeline += gp.PrintProfilingStats(every=profile_every)

    # Create a request
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(affs, output_size)
    request.add(seg, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(lsds, output_size)
    request.add(pred_lsds, output_size)

    comet_exp = comet_ml.get_running_experiment()
    
    # Build the pipline and train
    with gp.build(pipeline):
        for i in range(prev_iter, num_iterations+1):
            if not i%profile_every and i != 0:
                if comet_exp is not None:
                    # This message goes first so it is right after the profiling stats
                    comet_ml.logging.info(f'Experiment running: {url}')
                else:
                    logging.warning('\x1b[1;31m' + 'Logging to comet is disabled.' + '\x1b[0m')

            batch = pipeline.request_batch(request)     

            if comet_exp is not None:      
                url = '\x1b[36m' + comet_exp.url + '\x1b[0m' # url in blue because it looks fancy

                comet_log_batch(i, batch, request, loss_every=1, img_every=10000)

                if not i%save_every and i != 0:
                    comet_exp.log_model(comet_exp.project_name, 
                                        os.path.join(model_dir, f'model_checkpoint_{i}'))


if __name__ == '__main__':

    parser=argparse.ArgumentParser('')
    parser.add_argument('-p', '--projectdir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Absolute or relative path to the project destination dir.')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG',
                        dest='training_config',
                        required=True,
                        type=str,
                        help='Absolute or relative path to the training config JSON file.')
    parser.add_argument('--gpu-id',
                        metavar='GPU_ID',
                        dest='GPU_ID',
                        required=False,
                        nargs='+',
                        default=0,
                        type=int,
                        help='GPU PID to use for training. Default: 0')   
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=1,
                        help='Number of workers to use for prepping data for the GPU.\
                             Default: 1')
    parser.add_argument('-r', '--resume-training',
                        metavar='RESUME',
                        dest='resume_training',
                        type=str,
                        default=None,
                        help='Project to resume an existing training bout. Provide an experiment name found in project_dir, \
                            or set to -1 to resume latest training bout.')
    parser.add_argument('--no-comet-log',
                        dest='no_comet_log',
                        default=False,
                        action='store_true',
                        help='Disable logging to comet. Useful when testing changes to the script or ground-truth.')
    args=parser.parse_args()

    start_train(**vars(args))
