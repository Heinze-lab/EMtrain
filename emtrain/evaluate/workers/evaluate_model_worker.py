import json
import numpy as np
import sys

from emsegment.Segment import segment_dataset
from emsegment.FindSegments import find_segments
from emtrain.evaluate.evaluate_volume import EvaluateAnnotations
from pymongo import MongoClient


def evaluate_model(project_dir,
                   project_prefix,
                   model_config,
                   eval_config, 
                   catmaid_secrets,
                   seg_config_path,
                   num_workers,
                   thresholds_minmax,
                   thresholds_step,
                   GPU_pool):

    seg_config = segment_dataset(
                                project_dir=project_dir,
                                project_prefix=project_prefix,
                                model_config=model_config,
                                input_path=eval_config['raw_path'],
                                raw_dataset=eval_config['raw_dataset'],
                                GPU_pool=GPU_pool,
                                num_workers=num_workers,
                                volume_suffix=eval_config['volume_suffix'],
                                roi_start=eval_config['roi_start_nm'],
                                roi_size=eval_config['roi_size_nm'],
                                seg_config=seg_config_path,
                                continue_previous=True,
                                start_over=False,
                                return_config=True
                                )
        
    edges_collection = seg_config['agglo_config']['edges_collection_basename'] + '_' + seg_config['agglo_config']['merge_function']
    fragments_path = seg_config['fragments_path']
    fragments_dataset = seg_config['fragments_dataset']

    # Find connected components
    db = MongoClient(None)[seg_config['db_name']]
    if db.info_segmentation.find_one({'task': 'find_segments'}) is None:
        find_segments(
                    db_name=seg_config['db_name'],
                    fragments_path=fragments_path,
                    fragments_dataset=fragments_dataset,
                    edges_collection=edges_collection,
                    thresholds_minmax=thresholds_minmax,
                    thresholds_step=thresholds_step,
                    num_workers=num_workers)
    
    # Run evaluation
    eval = EvaluateAnnotations(fragments_file=fragments_path,
                               fragments_dataset=fragments_dataset,
                               db_host=None,
                               edges_db_name=seg_config['db_name'],
                               edges_collection=edges_collection,
                               catmaid_pid=eval_config['catmaid_project_id'],
                               test_sk_annotation=eval_config['test_sk_annotation'],
                               test_volume=eval_config['catmaid_volume_name'],
                               thresholds_minmax=thresholds_minmax,
                               thresholds_step=thresholds_step,
                               num_workers=num_workers,
                               catmaid_secrets=catmaid_secrets,
                               compute_mincut_metric=False)
    eval.evaluate()



if __name__ == '__main__':

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_model(**config)