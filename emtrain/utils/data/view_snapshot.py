import neuroglancer
from time import sleep
from funlib.persistence import open_ds
import sys
import os
import numpy as np

def data_to_LocalVolume(
                      data,
                      spatial_dims,
                      voxel_offset,
                      voxel_size
                       ):

    spatial_dim_names = ["t", "z", "y", "x"]
    channel_dim_names = ["b^", "c^"]

    dims = len(data.shape)
    channel_dims = dims - spatial_dims
    voxel_offset = [0] * channel_dims + list(voxel_offset)

    attrs = {
             "names": (channel_dim_names[-channel_dims:] if channel_dims > 0 else [])
             + spatial_dim_names[-spatial_dims:],
             "units": [""] * channel_dims + ["nm"] * spatial_dims,
             "scales": [1] * channel_dims + list(voxel_size),
            }

    dimensions = neuroglancer.CoordinateSpace(**attrs)

    local_volume = neuroglancer.LocalVolume(
                                            data=data,
                                            voxel_offset=voxel_offset,
                                            dimensions=dimensions
                                           )
    return local_volume


def view_snapshot(snapshot,
                  port=None):
    
    raw = open_ds(os.path.join(snapshot, 'raw'))
    voxel_size = raw.voxel_size
    raw = raw.data

    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=port)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers['raw'] = neuroglancer.ImageLayer(source=data_to_LocalVolume(raw, 
                                                                               3, 
                                                                               [0,0,0], 
                                                                               voxel_size))
        for dataset in ['raw', 'gt_seg', 'affs', 'lsds', 'pred_affs', 'pred_lsds']:
            ds = open_ds(os.path.join(snapshot, dataset))
            arr = ds.data
            
            shape_diff = np.array(raw.shape) - np.array(arr.shape[-3:])
            
            if shape_diff.any():
                offset = np.uint64(shape_diff/2)
            else:
                offset = [0,0,0]

            if dataset == 'gt_seg':
                s.layers[dataset] = neuroglancer.SegmentationLayer(source=data_to_LocalVolume(arr, 
                                                                                              3, 
                                                                                              offset, 
                                                                                              voxel_size))
            else:  
                s.layers[dataset] = neuroglancer.ImageLayer(source=data_to_LocalVolume(arr, 
                                                                                       3, 
                                                                                       offset,
                                                                                       voxel_size))


    url = viewer.get_viewer_url()
    print('http://localhost:' + url.split(':')[-1])

    sleep(1)

    input()

if __name__ == '__main__':

    view_snapshot(*sys.argv[1:])
