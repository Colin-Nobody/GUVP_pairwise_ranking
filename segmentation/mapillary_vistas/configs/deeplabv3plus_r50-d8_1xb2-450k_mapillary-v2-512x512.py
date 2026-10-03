"""Long low-learning-rate continuation of the Mapillary Vistas v2 run.

Resume from the trusted ``iter_96000.pth`` checkpoint.  Its optimizer,
sampler and PolyLR state are retained, including the established 1e-4 learning
rate floor used for the 80k-to-96k continuation.  ``max_iters`` is the total
iteration count, so this adds 354,000 iterations rather than restarting.
"""

_base_ = ['./deeplabv3plus_r50-d8_1xb2-96k_mapillary-v2-512x512.py']

train_cfg = dict(type='IterBasedTrainLoop', max_iters=450000, val_interval=4000)

# Keep the recent 96k validation/checkpoint cadence.  The work directory is
# intentionally separate so the completed 96k run remains immutable.
work_dir = (
    '/home/colin/GUVP_pairwise_ranking/output/'
    'deeplabv3plus_r50_mapillary_v2_450k'
)
load_from = None
