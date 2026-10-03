"""Continuation of the completed 80k Mapillary Vistas run.

It is intended to resume ``iter_80000.pth`` in the existing work directory.
The restored PolyLR has already reached its 1e-4 floor, so this 16k extension
tests whether low-learning-rate fine-tuning produces a meaningful gain.
"""

_base_ = ['./deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py']

# Validate twice as often during the extension to distinguish a true plateau
# from a final-checkpoint fluctuation.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=96000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5,
    ))

# ``--resume`` restores the optimizer, scheduler and iteration counter from
# iter_80000.pth in the work directory; do not load the ADE initializer.
load_from = None
