import os
import json
import logging
import numpy as np
import sys

from collections import defaultdict
from pymongo import MongoClient
from skopt import Optimizer
from skopt.space import Real, Integer

logging.basicConfig(level=logging.INFO)

def get_augment_parameters(output_path,
                           project_dir=None,
                           eval_config=None,
                           db_host=None,
                           mode='random',
                           return_config=False,
                           exploit=False):
    
    search_space = {'elastic_spacing': np.uint64([40, 101]),
                    'elastic_jitter': np.uint64([0, 11]),
                    'prob_elastic': np.float32([0.1, 0.7]),
                    'intensity_scmin': np.float32([0.5, 1]),
                    'intensity_scmax': np.float32([1, 1.5]),
                    'intensity_shmin': np.float32([-0.35, -0.05]),
                    'intensity_shmax': np.float32([0.05, 0.35]),
                    'prob_noise': np.float32([0.01, 0.1]),
                    'prob_missing': np.float32([0.01, 0.2]),
                    'prob_low_contrast': np.float32([0.01, 0.2]),
                    }
    
    logging.info('Computing augment parameters...')
    if mode == 'random':
        # Randomly select a combination of parameters within the search space
        augment_config = random_search(search_space)
    elif mode == 'bayesopt':
        # Use evaluations of previous models to choose the optimal next parameter combination to use
        augment_config = bayesian_opt(search_space,
                                      project_dir,
                                      eval_config,
                                      db_host,
                                      exploit=exploit)
    else:
        raise NotImplementedError(f'{mode} for augment parameters search not implemented')

    if return_config:
        return augment_config

    logging.info(f'Saving parameters at {output_path}')
    with open(output_path, 'w') as f:
        json.dump(augment_config, f, indent='')


def random_search(search_space):

    # gp.DeformAugment
    elastic_spacing = [10] + [np.random.randint(*search_space['elastic_spacing'])]*2
    elastic_jitter = [0] + [np.random.randint(*search_space['elastic_jitter'])]*2
    prob_elastic = np.random.uniform(*search_space['prob_elastic']) 

    # gp.IntensityAugment
    intensity_scmin = np.random.uniform(*search_space['intensity_scmin'])
    intensity_scmax = np.random.uniform(*search_space['intensity_scmax'])
    intensity_shmin = np.random.uniform(*search_space['intensity_shmin'])
    intensity_shmax = np.random.uniform(*search_space['intensity_shmax'])

    # gp.NoiseAugment
    prob_noise = np.random.uniform(*search_space['prob_noise'])

    # gp.DefectAugment
    prob_missing = np.random.uniform(*search_space['prob_missing'])
    prob_low_contrast = np.random.uniform(*search_space['prob_low_contrast'])
    
    augment_config =  {
                    'elastic_spacing': elastic_spacing,
                    'elastic_jitter': elastic_jitter,
                    'prob_elastic': prob_elastic,
                    'intensity_scmin': intensity_scmin,
                    'intensity_scmax': intensity_scmax,
                    'intensity_shmin': intensity_shmin,
                    'intensity_shmax': intensity_shmax,
                    'prob_noise': prob_noise,
                    'prob_missing': prob_missing,
                    'prob_low_contrast': prob_low_contrast,
                      }

    return augment_config


def bayesian_opt(search_space,
                 project_dir,
                 eval_config,
                 db_host=None,
                 exploit=False):
    
    '''
    project_dir: Directory containing the models. The name of the directory will be considered as the project name.
    '''
    
    with open(eval_config, 'r') as f:
        eval_config = json.load(f)
    
    # Currently uses VOI sum as the optimization parameter
    # Get search space in the skopt format so it does not throw warnings
    for k,v in search_space.items():
        if v.dtype == np.uint64:
            search_space[k] = Integer(*v)
        if v.dtype == np.float32:
            search_space[k] = Real(*v)
    
    # Find the relevant databases containing the models' evaluations
    project_name = os.path.basename(project_dir)
    volume_suffix = eval_config['volume_suffix']
    volume_name = os.path.basename(eval_config['raw_path']).rstrip('.zarr')
     
    client = MongoClient(db_host)
    db_names = []
    for d in client.list_database_names():
        if np.logical_and.reduce([pattern in d for pattern in ['eval', project_name, volume_suffix]]):
            db_names.append(d)

    # Get evaluations and corresponding augmentation parameters
    models_stats = defaultdict(dict)
    for db_name in db_names:
        db = client[db_name]

        edges_collection = db.info_segmentation.find_one({'task': 'agglomeration'})['edges_collection']
        coll_name = '_'.join(['eval', volume_name, edges_collection])
        coll = db[coll_name]
        docs = list(coll.find({},{'_id':0,
                                'voi_split': 1,
                                'voi_merge': 1
                                }))

        voi_split = np.array([d['voi_split'] for d in docs])
        voi_merge = np.array([d['voi_merge'] for d in docs])
        voi_sum = np.sum([voi_split, voi_merge], axis=0)   

        model_path = client[db_name].info_segmentation.find_one({'task': 'prediction'}, {'_id':0, 'model_path':1})['model_path']
        model_dir, _, checkpoint = model_path.rsplit('/', maxsplit=2)
        checkpoint = int(checkpoint.strip('model_checkpoint_'))
        model_name = model_dir.split('/')[-1]
        augment_config = os.path.join(model_dir, 'augment_config.json')

        with open(augment_config, 'r') as f:
            augment_config = json.load(f)

        models_stats[model_name][checkpoint] = voi_sum.min()
        models_stats[model_name]['hyperparameters'] = augment_config

    # Organize data
    hyperparameters = []
    scores = []
    for stats in models_stats.values():
        h = []
        for k in search_space.keys():
            p = stats['hyperparameters'][k]
            if isinstance(p, list):
                p = p[-1]
            h.append(p)

        score = min([stats[k] for k in stats.keys() if isinstance(k, int)])
        hyperparameters.append(h)
        scores.append(score)

    # Inform the optimizer
    # As far as I can tell, these are good parameters in our case
    # At the very least, they work in most cases
    if exploit:
        # Exploit areas with high probability to improve
        acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}
    else:
        # Explores more
        acq_func_kwargs = {"xi": 10000, "kappa": 10000}

    optimizer = Optimizer(list(search_space.values()), 
                          base_estimator='GP', 
                          acq_func='EI',
                          acq_func_kwargs=acq_func_kwargs)
    for p, score in zip(hyperparameters, scores):
        optimizer.tell(p, score)

    # Ask the optimizer for the next best hyperparameters
    # By default the optimizer tries to minimize the score
    next_hyperparameters = optimizer.ask()

    # Organize config and return
    augment_config = {}
    for k, p in zip(list(search_space.keys()), next_hyperparameters):
        if isinstance(p, np.int64):
            p = int(p)
        elif isinstance(p, np.float32):
            p = float(p)

        if k == 'elastic_spacing':
            p = [10, p, p]
        elif k == 'elastic_jitter':
            p = [0, p, p]

        augment_config[k] = p
    
    return augment_config


if __name__ == '__main__':
    
    output_path = sys.argv[1]
    get_augment_parameters(output_path)