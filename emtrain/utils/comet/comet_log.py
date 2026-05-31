import comet_ml
import numpy as np

from funlib.segment.arrays import relabel


def comet_log_batch(i, batch, request, loss_every=1, img_every=10000):
    '''
    Log batch info to comet.
    '''

    comet_exp = comet_ml.get_running_experiment()
    
    if not i%loss_every:
        comet_exp.log_metrics({'batch_loss':batch.loss}, step=i)

    if not i%img_every:
        comet_log_batch_images(i, batch, request)


def comet_log_batch_images(i, batch, request):
    '''
    Log images to comet.
    '''

    comet_exp = comet_ml.get_running_experiment()
    roi = request.get_common_roi()
    
    for key, arr in batch.arrays.items():
        if str(key) == 'AFFS_WEIGHTS':
            continue

        img = arr.crop(roi).data
        z = 4

        if str(key) == 'RAW':
            # Raw data
            img = img[z]
        elif str(key) == 'SEGMENTATION':
            # Ground-truth segmentation
            img = img[z]
            img = relabel(img)[0].astype(np.int32)
        elif str(key) in ['AFFINITIES', 'PRED_AFFINITIES']:
            # Ground-truth and prediction affinities
            img = img[:, z, ...]
        elif str(key) in ['LSDS', 'PRED_LSDS']:
            # Ground-truth and prediction lsds
            # Different channel ranges are saved separately
            # See https://localshapedescriptors.github.io/

            # offset center of mass
            lsd_0 = img[0:3, z, ...]
            # covariance (direction)
            lsd_1 = img[3:6, z, ...]
            # pearson's correlation coefficient
            lsd_2 = img[6:9, z, ...]
            # voxel count
            lsd_3 = img[9, z, ...]

            for c, img in enumerate([lsd_0, lsd_1, lsd_2, lsd_3]):
                comet_exp.log_image(img.T, 
                                    name=f'{str(key)}_{c}', 
                                    step=i)
            continue
        
        # Save image if not lsds
        comet_exp.log_image(img.T, 
                            name=str(key),
                            step=i)