"""DeepLabV3+ R50-D8: 450k Mapillary training with whole-image baseline resize.

Train and validation use the same deterministic geometric transform:
``Resize(scale=(2048, 512), keep_ratio=True)``.  Training retains only
appearance augmentation (flip and photometric distortion), never crops a
512x512 patch. Batch size is one so variable-width panoramas do not create a
batch-padding memory spike; image and labels are still padded to multiples of
32.
"""

_base_ = ['./deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py']

baseline_resize_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='UseResizedShapeForEvaluation'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=baseline_resize_pipeline),
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# Unlike fixed 512x512 patches, whole resized images have variable widths.
# Padding to 32 preserves encoder/decoder stride compatibility without
# altering the semantic content or turning labels into resized probabilities.
data_preprocessor = dict(size=None, size_divisor=32)
# The ASPP image-pooling branch is 1x1. GroupNorm, unlike BatchNorm, remains
# well-defined for batch=1; freezing backbone BN statistics also avoids
# unstable per-image updates with a one-image mini-batch.
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(with_cp=False, norm_eval=True),
)

# Base LR .02 is automatically scaled by 1/16 for batch=1: effective .00125.
optim_wrapper = dict(optimizer=dict(lr=0.02))
custom_imports = dict(
    imports=['mapillary_transforms', 'mapillary_batch1_hooks'],
    allow_failed_imports=False,
)
custom_hooks = [dict(type='FreezeASPPImagePoolNormHook')]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=450000, val_interval=4000)
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=450000, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=4000,
        save_best='mIoU', rule='greater', max_keep_ckpts=5,
    ),
)

work_dir = (
    '/home/colin/GUVP_pairwise_ranking/output/'
    'deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k'
)
