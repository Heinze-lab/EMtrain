import json
import logging
import os

from glob import glob

from emtrain.tuning.prep_augments import get_augment_parameters
from emtrain.evaluate.evaluate_model import evaluate_model_checkpoints


logging.basicConfig(level=logging.INFO)


def prep_training_experiment(experiment_dir,
                             training_config,
                             GPU_ID=None,
                             num_workers=None):
    
    '''
    experiment dir: Directory that will contain the model and configs. Its parent directory is the project dir
    '''

    with open(training_config, 'r') as f:
        training_config = json.load(f)

    ground_truth_config = training_config['ground_truth']
    model_config        = training_config['model']
    augment_config      = training_config['augment_config']
    eval_config         = training_config['eval_config']
    augment_path        = augment_config.get('path')

    with open(ground_truth_config['ground_truth_config'], 'r') as f:
        ground_truth_config = json.load(f)

    if augment_path is not None:
        logging.info('Using existing config files.')
        # This is the config of an existing experiment
        with open(augment_path) as f:
            augment_config = json.load(f)
        return training_config, ground_truth_config, model_config, augment_config
   
    logging.info('Computing config files for the new experiment.')
    # This is a new experiment, compute new augment parameters
    search_mode = augment_config['search_mode']
    project_dir = os.path.abspath(experiment_dir).rsplit('/', maxsplit=1)[0] # Parent directory containing all experiments
    if search_mode == 'bayesopt' and augment_path is None:
        exploit = augment_config['exploit']
        logging.info('Hyperparameters will be determined with bayesian optimization')
        logging.info(f'Exploitation: {exploit}')

        # Ensure that all models of this project have been evaluated before computing augments
        project_name = project_dir.rsplit('/')[-1]
        model_dirs = glob(os.path.join(project_dir, f'*{project_name}*/'))

        output_dir = os.path.join(project_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        for model_dir in model_dirs:
            evaluate_model_checkpoints(output_dir=output_dir,
                                       model_dir=model_dir,
                                       eval_config=eval_config['path'],
                                       num_workers=num_workers,
                                       GPU_pool=GPU_ID,
                                       checkpoints_start=eval_config['checkpoints_start'],
                                       checkpoints_end=eval_config['checkpoints_end'],
                                       seg_config_path='/mnt/hdd1/SRC/EMpipelines/EMtrain/emtrain/evaluate/seg_config.json'
                                       )
    else:
        logging.info('Hyperparameters will be determined randomly')
    
    # Compute the next augment parameters
    augment_config = get_augment_parameters(output_path=None,
                                            project_dir=project_dir,
                                            eval_config=eval_config['path'],
                                            db_host=None,
                                            mode=search_mode,
                                            return_config=True,
                                            exploit=exploit)

    # Create project dir if doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Save augment config
    augment_path = os.path.join(experiment_dir, 'augment_config.json')
    with open(augment_path, 'w') as f:
        json.dump(augment_config, f, indent='')

    # Save ground-truth config
    ground_truth_path = os.path.join(experiment_dir, 'ground_truth_config.json')
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth_config, f, indent='')

    # Save training config
    training_config['experiment_dir'] = experiment_dir
    training_config['augment_config']['path'] = augment_path
    training_config['ground_truth']['ground_truth_config'] = ground_truth_path
    
    training_path = os.path.join(experiment_dir, 'training_config.json')
    with open(training_path, 'w') as f:
        json.dump(training_config, f, indent='')

    return training_config, ground_truth_config, model_config, augment_config