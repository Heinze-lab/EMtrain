import comet_ml
import numpy as np

from funlib.segment.arrays import relabel


# Aliases — keep both the historical key names (used by old training runs
# that hardcoded gp.ArrayKey('PRED_AFFINITIES')) and the new ones produced
# by the model registry (`output_keys` -> uppercase).
AFF_KEYS    = {'AFFINITIES', 'PRED_AFFINITIES', 'PRED_AFFS'}
LSD_KEYS    = {'LSDS', 'PRED_LSDS'}
GP_KEYS     = {'GP_LOGIT', 'GP_UNCERTAINTY'}


def comet_log_batch(i, batch, request, loss_every=1, img_every=10000):
    '''Log batch info to comet.'''

    comet_exp = comet_ml.get_running_experiment()

    if not i % loss_every:
        comet_exp.log_metrics({'batch_loss': batch.loss}, step=i)

    if not i % img_every:
        comet_log_batch_images(i, batch, request)


def _mid_z(arr, z_axis):
    """Mid-slice index along ``z_axis`` of ``arr``."""
    return arr.shape[z_axis] // 2


def comet_log_batch_images(i, batch, request):
    '''Log midplane images to comet.'''

    comet_exp = comet_ml.get_running_experiment()
    roi = request.get_common_roi()

    for key, arr in batch.arrays.items():
        name = str(key)
        if name == 'AFFS_WEIGHTS':
            continue

        img = arr.crop(roi).data

        if name == 'RAW':
            # (D, H, W)
            img = img[_mid_z(img, 0)]

        elif name == 'SEGMENTATION':
            # (D, H, W)
            img = img[_mid_z(img, 0)]
            img = relabel(img)[0].astype(np.int32)

        elif name in AFF_KEYS:
            # (3, D, H, W) -> mid z, keep channels
            img = img[:, _mid_z(img, 1), ...]

        elif name in LSD_KEYS:
            # (10, D, H, W) -> 4 separate images, one per LSD group.
            # See https://localshapedescriptors.github.io/
            z = _mid_z(img, 1)
            lsd_groups = [
                img[0:3, z, ...],   # offset to center of mass
                img[3:6, z, ...],   # covariance (direction)
                img[6:9, z, ...],   # pearson's correlation
                img[9,   z, ...],   # voxel count
            ]
            for c, lsd_img in enumerate(lsd_groups):
                comet_exp.log_image(lsd_img.T, name=f'{name}_{c}', step=i)
            continue

        elif name in GP_KEYS:
            # (C, D', H', W') -> single-channel coarse heatmap.
            # GP head outputs num_classes (default 1) so C is usually 1; if
            # a multi-class config is ever used, log each channel separately.
            z = _mid_z(img, 1)
            for c in range(img.shape[0]):
                comet_exp.log_image(img[c, z].T,
                                    name=name if img.shape[0] == 1
                                    else f'{name}_{c}',
                                    step=i)
            continue

        else:
            # Unknown array — emit a best-effort midplane so we don't
            # crash a whole image-logging step on one stray key.
            if img.ndim == 3:
                img = img[_mid_z(img, 0)]
            elif img.ndim == 4:
                img = img[:, _mid_z(img, 1), ...]
            else:
                continue

        comet_exp.log_image(img.T, name=name, step=i)
