import argparse
import logging
import numpy as np
import os

from funlib.persistence import prepare_ds
from glob import glob
from tifffile import imread
from tqdm import tqdm
from zarr import Blosc

logging.basicConfig(level=logging.INFO)

def tifs_to_zarr(input_path,
                 voxel_size,
                 dataset,
                 output_path=None):

    input_path = os.path.abspath(input_path)
    
    if dataset == 'raw':
        dtype = np.uint8
    else:
        dtype = np.uint64

    # Import images
    if input_path.endswith('.tif'):
        image_paths = input_path
    else:
        image_paths = sorted(glob(input_path + '/*.tif'))

    image_stack = imread(image_paths).astype(dtype)
    logging.info(f'Found {image_stack.shape[0]} images')

    # Write to zarr
    if output_path is None:
        output_path = os.path.dirname(input_path)
        
        if input_path.endswith('.tif'):
            output_path += '/' + input_path.split('/')[-1][:-4] + '.zarr'
        else:
            output_path += '/' + os.path.basename(input_path) + '.zarr'

    output_path += '/' + dataset
    output_path = os.path.abspath(output_path)

    logging.info(f'Writing to {output_path}')
    ds = prepare_ds(output_path,
                    shape=image_stack.shape,
                    offset=[0,0,0],
                    voxel_size=voxel_size,
                    axis_names=['z', 'y', 'x'],
                    units=['nm']*3,
                    dtype=dtype,
                    compressor=Blosc('zlib', clevel=5))
    ds[...] = image_stack

    logging.info('Done!')

if __name__ == '__main__':

    parser=argparse.ArgumentParser('')
    parser.add_argument('-i', '--input_path',
                        metavar='INPUT_PATH',
                        dest='input_path',
                        required=True,
                        nargs='+',
                        type=str,
                        help='Path to tif file or input dir containing tifs.')
    parser.add_argument('-d', '--dataset',
                        metavar='DATASET',
                        dest='dataset',
                        required=True,
                        type=str,
                        help='Subdirectory of the zarr container to write to.')
    parser.add_argument('-v', '--voxel_size',
                        metavar='VOXEL_SIZE',
                        dest='voxel_size',
                        required=False,
                        nargs='+',
                        default=[50,10,10],
                        type=int,
                        help='Voxel size in nm (ZYX)') 
    parser.add_argument('-o', '--output_path',
                        metavar='OUTPUT_PATH',
                        dest='output_path',
                        required=False,
                        nargs='+',
                        type=str,
                        default=[None],
                        help='Path to zarr container where to write stack.')
    args=parser.parse_args()


    if len(args.input_path) > 1 and len(args.output_path) == 1:
        args.output_path = [None] * len(args.input_path)

    for input_path, output_path in zip(args.input_path, args.output_path):
        tifs_to_zarr(input_path,
                     args.voxel_size,
                     args.dataset,
                     output_path)
