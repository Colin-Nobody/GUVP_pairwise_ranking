"""DeepLabV3+ R50-D8 on Mapillary Vistas v2.0.

The architecture, 512x512 crop, augmentation, SGD/PolyLR policy, and 80k
iteration schedule intentionally follow the existing ADE20K baseline.  The
per-device batch is reduced to two and AMP is enabled for the local 8 GB GPU.
"""

_base_ = [
    '../../vendor/mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../vendor/mmsegmentation/configs/_base_/default_runtime.py',
    '../../vendor/mmsegmentation/configs/_base_/schedules/schedule_80k.py',
]

custom_imports = dict(imports=['mapillary_transforms'], allow_failed_imports=False)

dataset_type = 'MapillaryDataset_v2'
data_root = '/home/colin/datasets/mapillary_vistas'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='training/images',
            seg_map_path='training/v2.0/labels',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='validation/images',
            seg_map_path='validation/v2.0/labels',
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=124),
    auxiliary_head=dict(num_classes=124),
)

# Preserve the ADE20K SGD/PolyLR recipe. AMP is the only numerical change.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

randomness = dict(seed=0, deterministic=False)

# Start from the previous ADE20K model with only its incompatible 150-class
# classifier tensors removed. Set MAPILLARY_INIT='' to use ImageNet only.
load_from = (
    '/home/colin/GUVP_pairwise_ranking/segmentation/checkpoints/'
    'deeplabv3plus_r50-d8_ade20k_without_classifier.pth'
)
work_dir = (
    '/home/colin/GUVP_pairwise_ranking/output/'
    'deeplabv3plus_r50_mapillary_v2_80k'
)

# The base ADE20K recipe assumes global batch 16. MMEngine scales the learning
# rate to the actually launched global batch (two on the local single GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
