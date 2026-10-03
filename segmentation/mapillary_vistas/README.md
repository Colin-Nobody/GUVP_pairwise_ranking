# DeepLabV3+ R50 on Mapillary Vistas v2.0

This experiment transfers the existing ADE20K DeepLabV3+ R50-D8 checkpoint
to Mapillary Vistas v2.0. It intentionally keeps the ADE20K baseline's model,
512x512 crop, augmentation sequence, SGD + PolyLR optimizer, 80k iteration
schedule, and 8k validation/checkpoint interval. AMP and batch size two are
the local adaptations required by the 8 GB RTX 5060.

Validation is computed after the same aspect-preserving `(2048, 512)` resize,
with both image and categorical label at that resolution. This avoids
MMSegmentation allocating all 124 logit channels at the original multi-
megapixel Mapillary size. Target-domain audit masks are still restored to each
source image size after argmax, on CPU with nearest-neighbor interpolation.

## Data

The full ZIP is archived on the external disk. Only the semantic-training
subset is extracted to the Ubuntu internal filesystem for reliable training
I/O:

```text
/home/colin/datasets/mapillary_vistas/
  config_v2.0.json
  training/images/           # 18,000 JPEGs
  training/v2.0/labels/      # 18,000 indexed PNGs
  validation/images/         # 2,000 JPEGs
  validation/v2.0/labels/    # 2,000 indexed PNGs
```

The full ZIP remains at `/media/colin/白/datasets/mapillary_vistas/`; the
external disk is not required while training. A different extracted-data
location can be supplied with `MAPILLARY_ROOT`.

## Validate and train

```bash
cd /home/colin/GUVP_pairwise_ranking
/home/colin/miniconda3/envs/guvp-seg/bin/python \
  segmentation/mapillary_vistas/verify_dataset.py --quick

/home/colin/miniconda3/envs/guvp-seg/bin/python \
  segmentation/mapillary_vistas/prepare_init_checkpoint.py

bash segmentation/mapillary_vistas/train.sh
```

For the unattended end-to-end run (train, select the best checkpoint, then
execute the target-domain audit and build the review UI):

```bash
bash segmentation/mapillary_vistas/run_full_experiment.sh
```

Resume an interrupted run with:

```bash
bash segmentation/mapillary_vistas/train.sh --resume
```

## 450k continuation

The completed 96k run can be continued to a total of 450k iterations without
overwriting its checkpoints:

```bash
bash segmentation/mapillary_vistas/train_450k.sh
```

This restores `iter_96000.pth` including optimizer, sampler, and scheduler
state, then writes new checkpoints and logs to
`output/deeplabv3plus_r50_mapillary_v2_450k/`.  The continuation retains the
established `1e-4` PolyLR floor used during the 80k-to-96k fine-tuning stage.

The launcher sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` only for this training
process because PyTorch 2.6+ otherwise refuses MMEngine's locally generated
optimizer/message-hub objects. Resume only checkpoints created by this trusted
project.

Evaluate a checkpoint with:

```bash
bash segmentation/mapillary_vistas/test.sh \
  output/deeplabv3plus_r50_mapillary_v2_80k/best_mIoU_iter_*.pth
```

Run the controlled 30-image target-domain audit and build the browser review
page after training:

```bash
bash segmentation/mapillary_vistas/run_post_training_audit.sh \
  output/deeplabv3plus_r50_mapillary_v2_80k/best_mIoU_iter_*.pth
```

This regenerates the same deterministic six-images-per-city manifest used by
the ADE20K audit, runs baseline and 512/25%-overlap sliding inference, validates
the outputs, and writes the review UI to
`output/segmentation_mapillary_v2_audit/visual_review/index.html`.

Training output is written to
`output/deeplabv3plus_r50_mapillary_v2_80k/`. Training data, checkpoints, and
logs are all on the internal disk; the external disk keeps the recoverable
source archive.
