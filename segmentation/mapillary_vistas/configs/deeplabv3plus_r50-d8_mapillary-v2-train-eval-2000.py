"""Evaluation-only config: deterministic 2,000-image Mapillary train subset."""

_base_ = "/home/colin/GUVP_pairwise_ranking/output/deeplabv3plus_r50_mapillary_v2_450k/deeplabv3plus_r50-d8_1xb2-450k_mapillary-v2-512x512.py"

train_eval_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(2048, 512), keep_ratio=True),
    dict(type="UseResizedShapeForEvaluation"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MapillaryDataset_v2",
        data_root="/home/colin/datasets/mapillary_vistas",
        data_prefix=dict(
            img_path="training/images", seg_map_path="training/v2.0/labels"),
        ann_file="/home/colin/GUVP_pairwise_ranking/data/mapillary_vistas_train_eval_2000.txt",
        pipeline=train_eval_pipeline,
    ),
)
test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
