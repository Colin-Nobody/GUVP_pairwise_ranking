#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
PROJECT_DIR="$REPO_DIR/segmentation/mapillary_vistas"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-40k-mapillary-v2-finetune-3e-5.py"
CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k/best_mIoU_iter_136000.pth"
OUTPUT_DIR="$REPO_DIR/output/segmentation_expansion_1000_audit"

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint is unavailable: $CHECKPOINT" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR/baseline/masks" ]]; then
  echo "Refusing to overwrite existing audit masks in $OUTPUT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO_DIR/segmentation/prepare_expansion_1000_audit_manifest.py" \
  --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" "$REPO_DIR/segmentation/run_audit.py" \
  --manifest "$OUTPUT_DIR/audit_manifest.csv" \
  --output-dir "$OUTPUT_DIR" \
  --mode baseline \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT" \
  --expected-images 30 2>&1 | tee "$OUTPUT_DIR/run.log"
"$PYTHON_BIN" "$REPO_DIR/segmentation/create_random_segmentation_triptychs.py" \
  --run-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/visual_review" \
  --manifest "$OUTPUT_DIR/segmentation_manifest.csv" \
  --selection-csv "$OUTPUT_DIR/selected_30.csv" \
  --count 30

cat > "$OUTPUT_DIR/REVIEW_GUIDE.md" <<'EOF'
# 1000 图扩展样本：分割模型目标域复核

复核包包含 20 个城市各 1 张城市中位污染分数图，以及 10 张低/高分压力样本。
模型为项目现有的 Mapillary Vistas v2 DeepLabV3+ R50-D8 136k 微调 checkpoint；
推理采用单尺度全图、保持长宽比的 baseline 路径，掩码以最近邻方式还原到原图大小。

请在 `audit_ratings.csv` 中对重点类别填写 0（严重错误/不可用）、1（部分可用）、
2（总体可靠）或 NA（类别不存在/无法判断），并记录道路/人行道、杆件、标牌、
小目标、植被等类别的混淆，以及压缩、模糊、全景边缘等失败模式。

逐图原图、彩色类别掩码和叠加图位于 `visual_review/triptychs/`；图像索引表为
`visual_review/selected_30.csv`。这是人工质量关卡；完成审核后再批准对其余样本
提取特征。该样本抽样覆盖 20 城，但低/高分压力样本只覆盖其中 10 城。
EOF

echo "Audit review package: $OUTPUT_DIR"
