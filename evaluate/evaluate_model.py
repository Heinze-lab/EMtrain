import json
import numpy as np
import os

from glob import glob
from pymongo import MongoClient
from subprocess import check_call
from tqdm import tqdm

from emtrain.evaluate.workers.evaluate_model_worker import evaluate_model


def evaluate_model_checkpoints(output_dir,
                               model_dir,
                               eval_config, 
                               num_workers,
                               GPU_pool,
                               checkpoints_start=None,
                               checkpoints_end=None,
                               seg_config_path='seg_config.json',
                               catmaid_secrets=None):
    
    '''
    output_dir: Directory where the zarr container will be written
    eval_config: Path to an evaluation config JSON file
    '''
    
    catmaid_secrets = os.environ['CATMAID_CREDENTIALS'] if catmaid_secrets is None else catmaid_secrets
    
    thresholds_minmax = [0,1]
    thresholds_step = 0.05
    
    # Find model info
    model_dir = os.path.abspath(model_dir)
    project_name, model_name = model_dir.split('/')[-2:]

    with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
        training_config = json.load(f)

    # Create destination directory
    project_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Find relevant model checkpoints
    model_checkpoints = {}
    for checkpoint_path in glob(os.path.join(model_dir, 'checkpoints/model_checkpoint*')):
        checkpoint_it = int(checkpoint_path.split('_')[-1])

        if checkpoint_it >= checkpoints_start and checkpoint_it <= checkpoints_end:
            # Only keep checkpoints that are contained in the provided range
            model_checkpoints.update({checkpoint_it: checkpoint_path})
    
    # Find eval info including test volume ROI
    with open(eval_config, 'r') as f:
        eval_config = json.load(f)
    species_name        = eval_config['project']
    volume_suffix       = eval_config['volume_suffix']
    raw_path            = eval_config['raw_path']

    # Run segmentation of the test volume for each checkpoint
    padding = np.array(training_config['model']['input_shape']) - np.array(training_config['model']['output_shape'])
    model_configs = {i: {
                'model_path': checkpoint,
                'num_fmaps': training_config['model']['num_fmaps'],
                'output_shape': [40, 160, 160],
                'padding': padding.tolist()
                     } for i, checkpoint in dict(sorted(model_checkpoints.items())).items()}
    
    with open(seg_config_path, 'r') as f:
        seg_config = json.load(f)
    edges_collection = seg_config['agglo_config']['edges_collection_basename'] + '_' + seg_config['agglo_config']['merge_function']

    for checkpoint_it, model_config in tqdm(model_configs.items(), desc='Evaluating models'):
        project_prefix = '_'.join(['eval', model_name, str(checkpoint_it), species_name])
        db_name = os.path.basename(raw_path).rstrip('.zarr')
        db_name = project_prefix + '_' + db_name
        db_name = db_name + '_' + volume_suffix + '_00'

        db = MongoClient(None)[db_name]
        test = []
        for coll_name in db.list_collection_names():
            test.append(np.all([p in coll_name for p in ['eval', edges_collection]]))
        if np.any(test):
            # This checkpoint has been tested already, skip it
            continue

        config = {
                'project_dir': project_dir,
                'project_prefix': project_prefix,
                'model_config': model_config,
                'eval_config': eval_config,
                'catmaid_secrets': catmaid_secrets,
                'seg_config_path': seg_config_path,
                'num_workers': num_workers,
                'thresholds_minmax': thresholds_minmax,
                'thresholds_step': thresholds_step,
                'GPU_pool': GPU_pool
                 }

        worker_script = '/mnt/hdd1/SRC/EMpipelines/EMtrain/emtrain/evaluate/workers/evaluate_model_worker.py'
        config_dir = os.path.join(os.path.dirname(worker_script), 'worker_config')
        config_path = os.path.join(config_dir, db_name + '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent='')

        command = ['python', 
                   worker_script, 
                   config_path]
        check_call(' '.join(command), shell=True)


if __name__ == '__main__':
    evaluate_model_checkpoints(output_dir,
                                model_dir,
                                eval_config, 
                                num_workers,
                                GPU_pool,
                                checkpoints_start=checkpoints_start,
                                checkpoints_end=checkpoints_end,
                                seg_config_path=seg_config)