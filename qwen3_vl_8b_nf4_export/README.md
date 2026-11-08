# Qwen3-VL-8B-Instruct NF4 backup

This is a bitsandbytes NF4 double-quantized export of Qwen/Qwen3-VL-8B-Instruct at revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b, generated from the locally cached original model using the GUVP runtime settings. It is not a GUVP fine-tuned model. The GPU compute dtype was bfloat16. See export_provenance.json and reload_smoke_test.txt. Safetensors files are stored using Git LFS.

Install compatible PyTorch, CUDA, transformers and bitsandbytes, then load the directory with AutoProcessor and AutoModelForImageTextToText. GPU-only loading was verified with transformers 5.15.1, bitsandbytes 0.50.1 and PyTorch 2.11.0+cu128. Image inference succeeded after limiting image dimensions to 512 pixels; larger images may exceed available VRAM.

Upstream model card, provenance and license: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

The project owner reports the current GPU cannot run their Qwen3 inference workload. This backup's one-image, 512-pixel, 32-token smoke test does not establish that the original pairwise-ranking workload or larger inputs can run on the GPU.
