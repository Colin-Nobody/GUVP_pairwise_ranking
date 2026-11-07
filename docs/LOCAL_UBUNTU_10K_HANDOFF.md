# 10k V8 deployment handoff for Ubuntu

Updated 2026-11-07: the three deployment models were trained locally on the 10,000-image V8 VLM/TrueSkill labels and tested on an RTX 5060. The 10,000 source images were checked against the HPC manifest SHA-256 values. The label and cross-validation results are from commit `5843efc94274a05c342b54c515543b7fbe518b6e`.

The deployment uses the original DINOv3 ViT-B/16 checkpoint, with blocks 10 and 11 and an MLP head trained at edge 512, backbone LR 1e-5, eight epochs, for seeds 42, 137 and 2026. The output is the arithmetic mean of the three predictions. These are VLM-derived relative scores, not human ground truth. Three city-held-out CV repeats have MAE 0.9523, 0.9581 and 0.9498; the final full-data fit is not itself a generalization estimate.

The files `output/vit_vp_10k_v8/deployment/finetuned_last2_v8/{model.json,fit_config.json,seed_42.safetensors,seed_137.safetensors,seed_2026.safetensors}` are the deployment ensemble. The seed weights are Git LFS objects. The original `models/checkpoints/dinov3_vitb.safetensors` is also required; its SHA-256 is `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`. The predictor validates the original and seed hashes against `model.json` and requires CUDA. The full ensemble was tested on an RTX 5060: one image and a three-image folder both passed, and the repeated image gave the same prediction. Peak CUDA allocated memory was 0.994 GiB.

After installing Git LFS and retrieving the real objects, run from the repository root in a CUDA-capable environment:

```bash
conda run -n guvp-vit python vit_vp/predict_finetuned.py /path/to/image_or_directory \
  --model-dir output/vit_vp_10k_v8/deployment/finetuned_last2_v8 \
  --checkpoint models/checkpoints/dinov3_vitb.safetensors \
  --output output/local_10k_predictions.csv
```

Specify `--model-dir`: the predictor's default points to an older model. Do not use LFS pointer files as weights. The 12.96 GB frozen-prefix cache was needed only for training and is not required for inference. HPC job 384732 was pending when the earlier handoff was written; the status of that separate job is unknown and is not evidence for these locally trained models.
