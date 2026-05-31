import daisy
import glob
import json
import logging
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import pymaid
import sys
import time
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate
from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi, \
        expected_run_length, \
        get_skeleton_lengths
from funlib.evaluate.split_merge import split_graph
from pymongo import MongoClient


logging.basicConfig(level = logging.INFO)


class EvaluateAnnotations():

    def __init__(
                 self,
                 fragments_file,
                 fragments_dataset,
                 db_host,
                 edges_db_name,
                 edges_collection,
                 catmaid_pid,
                 test_sk_annotation,
                 test_volume,
                 thresholds_minmax,
                 thresholds_step,
                 num_workers,
                 compute_mincut_metric,
                 catmaid_secrets,
                 annotations_synapses_collection_name = None,
                 iteration = None,
                 **kwargs):

        self.fragments_file = fragments_file
        self.fragments_dataset = fragments_dataset
        
        self.fragments = open_ds(os.path.join(self.fragments_file,self.fragments_dataset))
        self.db_host = db_host
        self.edges_db_name = edges_db_name
        self.edges_collection = edges_collection
        self.scores_db_name = edges_db_name
        self.scores_collection_name = '_'.join(['eval', test_volume, edges_collection])
        
        self.test_sk_annotation = test_sk_annotation 
        self.test_volume = test_volume        
        self.annotations_db_name = test_volume
        self.annotations_skeletons_collection_name = test_volume

        self.iteration = str(iteration)
        self.num_workers = num_workers
        self.annotations_synapses_collection_name = annotations_synapses_collection_name

        self.roi = self.fragments.roi
        self.thresholds_minmax = thresholds_minmax
        self.thresholds_step = thresholds_step
        self.compute_mincut_metric = compute_mincut_metric

        with open(catmaid_secrets, 'r') as f:
            catmaid_secrets = json.load(f)
        self.catmaid_client = pymaid.CatmaidInstance(server=catmaid_secrets['server'], 
                                                     api_token=catmaid_secrets['token'],
                                                     http_user=catmaid_secrets['http_user'],
                                                     http_password=catmaid_secrets['http_password'],
                                                     project_id=catmaid_pid,
                                                     caching=True)

        if iteration is not None:
            self.site_fragment_lut_directory = os.path.join(
                self.fragments_file,
                'luts/site_fragment_{iteration}')
        else:
            self.site_fragment_lut_directory = os.path.join(
                self.fragments_file,
                'luts/site_fragment')
        
        logging.info('Path to site fragment luts: '
                f'{self.site_fragment_lut_directory}')



    def store_lut_in_block(self, block):

        logging.info(f'Finding fragment IDs in block {block}')

        # Get all skeletons nodes
        #client = mongo_proxy.MongoProxy(
        #      MongoClient(self.db_host))  # Handles Auto-reconnect errors
        # client = MongoClient(self.db_host)
        # database = client[self.annotations_db_name]
        # skeletons_collection = \
        #     database[self.annotations_skeletons_collection_name + '.nodes']

        # bz, by, bx = block.read_roi.get_begin()
        # ez, ey, ex = block.read_roi.get_end()

        site_nodes = []
        for node, data in self.skeletons.nodes(data=True):
            coords = Coordinate((int(data['z']), 
                                 int(data['y']), 
                                 int(data['x'])
                                 ))

            if block.read_roi.contains(coords):
                site_nodes.append({'id': node,'z': coords[0], 'y': coords[1], 'x': coords[2]})

        site_fragment_lut, num_bg_sites = get_site_fragment_lut(
            self.fragments,
            site_nodes,
            block.write_roi)

        if site_fragment_lut is None:
            return

        # Store LUT
        block_lut_path = os.path.join(
            self.site_fragment_lut_directory,
            f'seg_{self.edges_collection}_{str(block.block_id[1])}')

        np.savez_compressed(
            block_lut_path,
            site_fragment_lut = site_fragment_lut,
            num_bg_sites = num_bg_sites)


    def prepare_for_roi(self):

        logging.info(f'Preparing evaluation for ROI {self.roi}...')

        self.read_skeletons()

        # Array with site IDs
        self.site_ids = np.array([n for n in self.skeletons.nodes()], 
                                 dtype=np.uint64)

        # Array with component ID for each site
        self.site_component_ids = np.array([data['component_id'] for _, data in self.skeletons.nodes(data=True)])

        assert self.site_component_ids.min() >= 0

        self.site_component_ids = self.site_component_ids.astype(np.uint64)
        self.number_of_components = np.unique(self.site_component_ids).size

        logging.info('Calculating skeleton lengths...')
        start = time.time()
        self.skeleton_lengths = get_skeleton_lengths(
                self.skeletons,
                skeleton_position_attributes=['z', 'y', 'x'],
                skeleton_id_attribute='component_id',
                store_edge_length='length')

        self.total_length = np.sum([l for _, l in self.skeleton_lengths.items()])


    def prepare_for_fragments(self):
        '''Get the fragment ID for each site in site_ids.'''

        logging.info(f'Preparing evaluation for fragments in '
                f'{self.fragments_file}...')

        if not os.path.exists(self.site_fragment_lut_directory):

            logging.info('site-fragment LUT does not exist, creating it...')

            os.makedirs(self.site_fragment_lut_directory)
            task = daisy.Task('frag_prep',
                              self.roi,
                              Roi((0, 0, 0), (2000, 2000, 2000)),
                              Roi((0, 0, 0), (2000, 2000, 2000)),
                              lambda b: self.store_lut_in_block(b),
                              num_workers = self.num_workers,
                              fit='shrink')
            daisy.run_blockwise([task])

        else:

            logging.info('site-fragment LUT already exists, skipping preparation')

        logging.info('Reading site-fragment LUTs from '
                f'{self.site_fragment_lut_directory}...')

        lut_files = glob.glob(
            os.path.join(
                self.site_fragment_lut_directory,
                '*.npz'))

        site_fragment_lut = np.concatenate(
            [
                np.load(f)['site_fragment_lut']
                for f in lut_files
            ],
            axis=1)

        self.num_bg_sites = int(np.sum([np.load(f)['num_bg_sites'] for f in lut_files]))

        assert site_fragment_lut.dtype == np.uint64

        logging.info(f'Found {len(site_fragment_lut[0])} sites in site-fragment LUT')

        # Convert to dictionary
        site_fragment_lut = {
            site: fragment
            for site, fragment in zip(
                site_fragment_lut[0],
                site_fragment_lut[1])
        }

        # Create fragment ID array congruent to site_ids
        self.site_fragment_ids = np.array([
            site_fragment_lut[s] if s in site_fragment_lut else 0
            for s in self.site_ids
        ], dtype=np.uint64)


    def read_skeletons(self):

        # Get all skeletons
        logging.info('Fetching all skeletons...')
        # skeletons_provider = daisy.persistence.MongoDbGraphProvider(
        #     self.annotations_db_name,
        #     self.db_host,
        #     nodes_collection = self.annotations_skeletons_collection_name +
        #     '.nodes',
        #     edges_collection = self.annotations_skeletons_collection_name +
        #     '.edges',
        #     endpoint_names = ['source', 'target'],
        #     position_attribute = ['z', 'y', 'x'],
        #     node_attribute_collections = {
        #                                   self.node_components: ['component_id']})

        # skeletons = skeletons_provider.get_graph(
        #         self.roi)

        skeletons = get_test_skeletons(self.test_sk_annotation, self.test_volume)

        logging.info(f'Found {skeletons.number_of_nodes()} skeleton nodes')

        # Remove outside edges and nodes
        remove_nodes = []
        for node, data in skeletons.nodes(data=True):
            coords = Coordinate((int(data['z']), 
                                 int(data['y']), 
                                 int(data['x'])))

            if not self.roi.contains(coords):
                remove_nodes.append(node)

        logging.info(f'Removing {len(remove_nodes)} nodes that were outside of ROI')

        for node in remove_nodes:
            skeletons.remove_node(node)
        
        self.skeletons = skeletons
        return True


    def evaluate(self):

        self.prepare_for_roi()

        self.prepare_for_fragments()

        thresholds = [round(i,2) for i in np.arange(
            float(self.thresholds_minmax[0]),
            float(self.thresholds_minmax[1]),
            self.thresholds_step)]

        procs = []

        logging.info('Evaluating thresholds...')

        for threshold in thresholds:
            proc = mp.Process(
                target = lambda: self.evaluate_threshold(threshold)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        return True


    def get_site_segment_ids(self, threshold):

        # Get fragment-segment LUT
        logging.info('Reading fragment-segment LUT...')
        start = time.time()

        fragment_segment_lut_dir = os.path.join(
                self.fragments_file,
                'luts',
                f'fragment_segment_{self.edges_collection}')

        logging.info('Reading fragment segment luts from: '
                f'{fragment_segment_lut_dir}')

        fragment_segment_lut_file = os.path.join(
                fragment_segment_lut_dir,
                'seg_%s_%d.npz' % (self.edges_collection, int(threshold*100)))

        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

        assert fragment_segment_lut.dtype == np.uint64

        # Get the segment ID for each site
        logging.info('Mapping sites to segments...')

        site_mask = np.isin(fragment_segment_lut[0], self.site_fragment_ids)
        site_segment_ids = replace_values(
            self.site_fragment_ids,
            fragment_segment_lut[0][site_mask],
            fragment_segment_lut[1][site_mask])

        return site_segment_ids, fragment_segment_lut


    def compute_expected_run_length(self, site_segment_ids):

        logging.info('Calculating expected run length...')

        node_segment_lut = {
            site: segment for site, segment in zip(
                self.site_ids,
                site_segment_ids)
        }

        erl, stats = expected_run_length(
                skeletons = self.skeletons,
                skeleton_id_attribute = 'component_id',
                edge_length_attribute = 'length',
                node_segment_lut = node_segment_lut,
                skeleton_lengths = self.skeleton_lengths,
                return_merge_split_stats = True)

        perfect_lut = {
                node: data['component_id'] for node, data in \
                        self.skeletons.nodes(data=True)
        }

        max_erl, _ = expected_run_length(
                skeletons = self.skeletons,
                skeleton_id_attribute = 'component_id',
                edge_length_attribute = 'length',
                node_segment_lut = perfect_lut,
                skeleton_lengths = self.skeleton_lengths,
                return_merge_split_stats = True)

        split_stats = [
            {
                'comp_id': int(comp_id),
                'seg_ids': [(int(a), int(b)) for a, b in seg_ids]
            }
            for comp_id, seg_ids in stats['split_stats'].items()
        ]

        merge_stats = [
            {
                'seg_id': int(seg_id),
                'comp_ids': [int(comp_id) for comp_id in comp_ids]
            }
            for seg_id, comp_ids in stats['merge_stats'].items()
        ]

        return erl, max_erl, split_stats, merge_stats


    def compute_splits_merges_needed(
            self,
            fragment_segment_lut,
            site_segment_ids,
            split_stats,
            merge_stats,
            threshold):

        total_splits_needed = 0
        total_additional_merges_needed = 0
        total_unsplittable_fragments = []

        logging.info('Computing min-cut metric for each merging segment...')

        for i, merge in enumerate(merge_stats):

            logging.info(f'Processing merge {i+1}/{len(merge_stats)}...')
            (
                splits_needed,
                additional_merges_needed,
                unsplittable_fragments) = self.mincut_metric(
                    fragment_segment_lut,
                    site_segment_ids,
                    merge['seg_id'],
                    merge['comp_ids'],
                    threshold)
            total_splits_needed += splits_needed
            total_additional_merges_needed += additional_merges_needed
            total_unsplittable_fragments += unsplittable_fragments

        total_merges_needed = 0
        for split in split_stats:
            total_merges_needed += len(split['seg_ids']) - 1
        total_merges_needed += total_additional_merges_needed

        return (
            total_splits_needed,
            total_merges_needed,
            total_unsplittable_fragments)


    def mincut_metric(
            self,
            fragment_segment_lut,
            site_segment_ids,
            segment_id,
            component_ids,
            threshold):

        # Get RAG for segment ID
        rag = self.get_segment_rag(segment_id, fragment_segment_lut, threshold)

        logging.info('Preparing RAG for split_graph call')

        # Replace merge_score with weight
        for _, _, data in rag.edges(data=True):
            data['weight'] = 1.0 - data['merge_score']

        # Find fragments for each component in segment_id
        component_fragments = {}

        # True for every site that maps to segment_id
        segment_mask = site_segment_ids == segment_id

        for component_id in component_ids:

            # limit following to sites that are part of component_id and
            # segment_id

            component_mask = self.site_component_ids == component_id
            fg_mask = self.site_fragment_ids != 0
            mask = np.logical_and(np.logical_and(component_mask, segment_mask), fg_mask)
            site_ids = self.site_ids[mask]
            site_fragment_ids = self.site_fragment_ids[mask]

            component_fragments[component_id] = site_fragment_ids

            for site_id, fragment_id in zip(site_ids, site_fragment_ids):

                if fragment_id == 0:
                    continue

                # For each fragment containing a site, we need a position for
                # the split_graph call. We just take the position of the
                # skeleton node that maps to it, if there are several, we take
                # the last one.

                site_data = self.skeletons.nodes[site_id]
                fragment = rag.nodes[fragment_id]
                fragment['z'] = site_data['z']
                fragment['y'] = site_data['y']
                fragment['x'] = site_data['x']

                # Keep track of how many components share a fragment. If it is
                # more than one, this fragment is unsplittable.
                if 'component_ids' not in fragment:
                    fragment['component_ids'] = set()
                fragment['component_ids'].add(component_id)

        # Find all unsplittable fragments...
        unsplittable_fragments = []
        for fragment_id, data in rag.nodes(data=True):
            if fragment_id == 0:
                continue
            if 'component_ids' in data and len(data['component_ids']) > 1:
                unsplittable_fragments.append(fragment_id)

        # ...and remove them from the component lists
        for component_id in component_ids:

            fragment_ids = component_fragments[component_id]
            valid_mask = np.logical_not(
                np.isin(
                    fragment_ids,
                    unsplittable_fragments))
            valid_fragment_ids = fragment_ids[valid_mask]
            if len(valid_fragment_ids) > 0:
                component_fragments[component_id] = valid_fragment_ids
            else:
                del component_fragments[component_id]

        logging.info(f'{len(unsplittable_fragments)} fragments are merging '
                'and can not be split')

        if len(component_fragments) <= 1:
            logging.info(
                'after removing unsplittable fragments, there is nothing to '
                'do anymore')
            return 0, 0, unsplittable_fragments

        # These are the fragments that need to be split
        split_fragments = list(component_fragments.values())

        logging.info(f'Splitting segment into {len(split_fragments)} components '
                f'with sizes {[len(c) for c in split_fragments]}')

        logging.info('Calling split_graph...')

        # Call split_graph
        num_splits_needed = split_graph(
            rag,
            split_fragments,
            position_attributes = ['z', 'y', 'x'],
            weight_attribute = 'weight',
            split_attribute = 'split')

        logging.info(f'{num_splits_needed} splits needed for segment '
                f'{segment_id}')

        # Get number of additional merges needed after splitting the current
        # segment
        #
        # this is the number of split labels per component minus 1
        additional_merges_needed = 0
        for component, fragments in component_fragments.items():
            split_ids = np.unique([rag.nodes[f]['split'] for f in fragments])
            additional_merges_needed += len(split_ids) - 1

        logging.info(f'{additional_merges_needed} additional merges '
                'needed to join components again')

        return (
            num_splits_needed,
            additional_merges_needed,
            unsplittable_fragments)


    def get_segment_rag(self, segment_id, fragment_segment_lut, threshold):

        logging.info(f'Reading RAG for segment {segment_id}')

        # rag_provider = daisy.persistence.MongoDbGraphProvider(
        #     self.edges_db_name,
        #     self.db_host,
        #     mode = 'r',
        #     edges_collection = self.edges_collection,
        #     position_attribute = ['z', 'y', 'x'])

        # Get all fragments for the given segment
        segment_mask = fragment_segment_lut[1] == segment_id
        fragment_ids = fragment_segment_lut[0][segment_mask]

        # Get the RAG containing all fragments
        nodes = [
            {'id': fragment_id, 'segment_id': segment_id}
            for fragment_id in fragment_ids
        ]
        
        client = MongoClient(self.db_host)
        edges_coll = client[self.edges_db_name][self.edges_collection]
            
        edges = []
        if len(fragment_ids) > 0:
            query_limit = 100000  # avoid BSON size limit
            num_chunks = (len(fragment_ids) - 1) // query_limit + 1
            query_chunks = np.array_split(fragment_ids, num_chunks)

            for f_ids in query_chunks:
                edges += list(edges_coll.find({'$or': [{'u':{'$in': f_ids.astype(int).tolist()}}, 
                                                        {'v':{'$in': f_ids.astype(int).tolist()}}]},
                                            {'_id': 0, 
                                            'u': 1,
                                            'v': 1,
                                            'merge_score': 1}))
            
        logging.info(f'RAG contains {len(nodes)} nodes & {len(edges)} edges')

        rag = nx.Graph()

        node_list = [
            (n['id'], {'segment_id': n['segment_id']})
            for n in nodes
        ]

        edge_list = [
            (e['u'], e['v'], {'merge_score': e['merge_score']})
            for e in edges
            if e['merge_score'] is not None
            if e['merge_score'] <= threshold
        ]

        rag.add_nodes_from(node_list)
        rag.add_edges_from(edge_list)

        rag.remove_nodes_from([
            n
            for n, data in rag.nodes(data=True)
            if 'segment_id' not in data])

        logging.info('after filtering dangling nodes and unmerged edges, RAG '
                f'contains {rag.number_of_nodes()} nodes & '
                f'{rag.number_of_edges()} edges')

        return rag


    def compute_rand_voi(
            self,
            site_component_ids,
            site_segment_ids,
            return_cluster_scores=False):

        logging.info('Computing RAND and VOI...')

        rand_voi_report = rand_voi(
            np.array([[site_component_ids]]),
            np.array([[site_segment_ids]]),
            return_cluster_scores=return_cluster_scores)

        logging.info(f'VOI split: {rand_voi_report["voi_split"]}')
        logging.info(f'VOI merge: {rand_voi_report["voi_merge"]}')

        return rand_voi_report


    def evaluate_threshold(self, threshold):

        #scores_client = mongo_proxy.MongoProxy(
        #        MongoClient(self.db_host))
        scores_client = MongoClient(self.db_host)
        scores_db = scores_client[self.scores_db_name]
        scores_collection = scores_db[self.scores_collection_name]

        site_segment_ids, fragment_segment_lut = \
            self.get_site_segment_ids(threshold)

        number_of_segments = np.unique(site_segment_ids).size

        erl, max_erl, split_stats, merge_stats = self.compute_expected_run_length(site_segment_ids)

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        logging.info(f'ERL: {erl}')
        logging.info(f'Max ERL: {max_erl}')
        logging.info(f'Total path length: {self.total_length}')

        normalized_erl = erl/max_erl
        logging.info(f'Normalized ERL: {normalized_erl}')

        if self.compute_mincut_metric:

            splits_needed, merges_needed, unsplittable_fragments = \
                self.compute_splits_merges_needed(
                    fragment_segment_lut,
                    site_segment_ids,
                    split_stats,
                    merge_stats,
                    threshold)

            average_splits_needed = splits_needed/number_of_segments
            average_merges_needed = merges_needed/self.number_of_components

            logging.info(f'Number of splits needed: {splits_needed}')
            logging.info(f'Number of background sites: {self.num_bg_sites}')
            logging.info(f'Average splits needed: {average_splits_needed}')
            logging.info(f'Average merges needed: {average_merges_needed}')
            logging.info('Number of unsplittable fragments: '
                    f'{len(unsplittable_fragments)}')

        rand_voi_report = self.compute_rand_voi(
            self.site_component_ids,
            site_segment_ids,
            return_cluster_scores=True)

        if self.annotations_synapses_collection_name:
            synapse_rand_voi_report = self.compute_rand_voi(
                self.site_component_ids[self.synaptic_sites_mask],
                site_segment_ids[self.synaptic_sites_mask])

        report = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del report[k]

        report['expected_run_length'] = float(erl)
        report['max_erl'] = float(max_erl)
        report['total path length'] = float(self.total_length)
        report['normalized_erl'] = float(normalized_erl)
        report['number_of_segments'] = number_of_segments
        report['number_of_components'] = self.number_of_components
        report['number_of_merging_segments'] = number_of_merging_segments
        report['number_of_split_skeletons'] = number_of_split_skeletons

        if self.compute_mincut_metric:
            report['total_splits_needed_to_fix_merges'] = splits_needed
            report['average_splits_needed_to_fix_merges'] = average_splits_needed
            report['total_merges_needed_to_fix_splits'] = merges_needed
            report['average_merges_needed_to_fix_splits'] = average_merges_needed
            report['number_of_unsplittable_fragments'] = len(unsplittable_fragments)
            report['number_of_background_sites'] = self.num_bg_sites
            report['unsplittable_fragments'] = [int(f) for f in unsplittable_fragments]

        report['merge_stats'] = merge_stats
        report['split_stats'] = split_stats
        report['threshold'] = float(threshold)
        #report['setup'] = self.setup
        report['iteration'] = self.iteration
        report['network_configuration'] = self.edges_db_name
        report['merge_function'] = self.edges_collection.strip('edges_')

        scores_collection.replace_one(
            filter={
                'network_configuration': report['network_configuration'],
                'merge_function': report['merge_function'],
                'threshold': report['threshold'],
            },
            replacement = report,
            upsert = True)

        find_worst_split_merges(rand_voi_report)


def get_site_fragment_lut(fragments, sites, roi):
    # Get the fragment IDs of all the sites that are contained in the given ROI

    sites = list(sites)

    if len(sites) == 0:
        logging.info(f'No sites in {roi}, skipping')
        return None, None

    logging.info(f'Getting fragment IDs for {len(sites)} synaptic sites in {roi}...')

    # For a few sites, direct lookup is faster than memory copies
    # if len(sites) >= 15:

    #     logging.info('Copying fragments into memory...')
    #     fragments = fragments[roi]

    logging.info(f'Getting fragment IDs for synaptic sites in {roi}...')

    fragment_ids = np.array([
        fragments[Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]
    site_ids = site_ids[fg_mask]

    lut = np.array([site_ids, fragment_ids])

    return lut, (fg_mask==0).sum()


def find_worst_split_merges(rand_voi_report):

    # Get most severe splits/merges
    splits = sorted([
        (s, i)
        for (i, s) in rand_voi_report['voi_split_i'].items()
    ])
    merges = sorted([
        (s, j)
        for (j, s) in rand_voi_report['voi_merge_j'].items()
    ])

    logging.info('10 worst splits:')
    for (s, i) in splits[-10:]:
        logging.info(f'\tcomponent {i}\tVOI split {s}')

    logging.info('10 worst merges:')
    for (s, i) in merges[-10:]:
        logging.info(f'\tsegment {i}\tVOI merge {s}')


def get_test_skeletons(annotation, vol_name, as_graph=True):
    vol = pymaid.get_volume(vol_name)

    neurons = pymaid.find_neurons(annotations=annotation, volumes=vol, intersect=True)
    pruned = neurons.prune_by_volume(vol)

    if as_graph:
        G = nx.Graph()

        for neuron in pruned:
            nodes = [(n['node_id'], {'component_id': int(neuron.skeleton_id),
                                     'z': int(n['z']),
                                     'y': int(n['y']),
                                     'x': int(n['x']),
                         }) for _, n in neuron.nodes.iterrows()]

            G.add_nodes_from(nodes)
            G.add_edges_from(neuron.edges)

        return G
    else:
        return pruned


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate = EvaluateAnnotations(**config)
    evaluate.evaluate()