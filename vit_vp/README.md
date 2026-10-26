# DINOv3 ViT→VP experiment

This directory contains the auditable image inventory, fixed nested city splits, frozen DINOv3 feature extractor, and first nested regression comparison. Read `../VIT_VP_EXPERIMENT_PLAN.md` and `../CODEX_CONTEXT.md` before changing study design or continuing a run.

The active environment is `guvp-vit`, created as an isolated clone of the existing `guvp-vlm` conda environment so the active VLM installation is untouched. It currently uses Python 3.11, PyTorch 2.11.0+cu128, Transformers 5.15.1, Safetensors 0.8.0, Hugging Face Hub 1.28.0, Pillow 12.3.0, and NumPy 2.4.6. Check the run metadata and environment lock under `output/vit_vp_1000_v1/` for the complete installed package record.

To recreate this workstation environment before the lock is refreshed, use:

```bash
conda create -n guvp-vit --clone guvp-vlm
```

The ViT checkpoint is a Safetensors file from `ZhengGuangze/dinov3`, pinned to repository revision `ec54a6b69c8ff6066571dec86ee2b7dd26836570`. Its SHA-256 is `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`. The model is instantiated using the DINOv3 ViT implementation in the installed Transformers package and strict state-dict loading after translating the reference checkpoint's `layer.*` prefix to `model.layer.*`.

Scripts are explicit and can be run from the repository root:

```bash
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/audit_dataset.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/make_splits.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/extract_features.py --max-edge 512
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/extract_features.py --max-edge 384
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/fit_frozen_ridge.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/fit_frozen_mlp.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/make_spatial_splits.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/fit_spatial_ridge.py --protocol 250m
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/fit_spatial_ridge.py --protocol buffer100
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/fit_edge_isolated.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/train_final.py
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/predict.py path/to/image.jpg
```

The nested city and spatial fits compare frozen ViT features with semantic Elastic Net using shared partitions. `fit_edge_isolated.py` recomputes training targets from within-training comparison edges and evaluates ranking only on held-out edges. `train_final.py` tunes on city-grouped folds and fits the deployment head on all 1,000 images after evaluation; `predict.py` accepts either an image or a directory. This all-data model's training fit is not a generalization estimate. Results and artifacts go only under the versioned `output/vit_vp_1000_v1/` namespace.

The Mapillary Vistas raw training and validation set is archived at `/media/colin/白/GUVP_archive/mapillary_vistas`. The original `/home/colin/datasets/mapillary_vistas` path is a symlink for legacy segmentation configs, so the external data drive must be mounted for those workflows. The ViT→VP experiment uses cached images/features and does not read the Mapillary raw set.
