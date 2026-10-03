"""140k low-learning-rate fine-tune from the best 450k Mapillary checkpoint."""

_base_ = ['./deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py']

# This is a new fine-tuning run: load model weights only, reset the optimizer,
# and avoid replaying the old InfiniteSampler position.
load_from = (
    '/home/colin/GUVP_pairwise_ranking/output/'
    'deeplabv3plus_r50_mapillary_v2_450k/best_mIoU_iter_432000.pth'
)

# Disable automatic global-batch scaling so 3e-5 is the effective LR.
auto_scale_lr = dict(enable=False, base_batch_size=16)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=3e-5, momentum=0.9, weight_decay=5e-4),
    clip_grad=None,
)
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=0, end=140000, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=140000, val_interval=2000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5,
    ))

work_dir = (
    '/home/colin/GUVP_pairwise_ranking/output/'
    'deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k'
)
