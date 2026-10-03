#!/usr/bin/env python3
"""Validate audit artifacts and write a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / 'output/segmentation_mapillary_v2_audit'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def validate_mode(audit: Path, mode: str, image_count: int, class_count: int) -> dict:
    mode_dir = audit / mode
    masks = list((mode_dir / 'masks').glob('*.png'))
    overlays = list((mode_dir / 'overlays').glob('*.jpg'))
    long_frame = pd.read_csv(mode_dir / 'class_counts_and_ratios.csv')
    summary = pd.read_csv(mode_dir / 'semantic_features_raw.csv')
    if len(masks) != image_count or len(overlays) != image_count:
        raise RuntimeError(f'{mode}: masks={len(masks)}, overlays={len(overlays)}')
    if len(long_frame) != image_count * class_count or len(summary) != image_count:
        raise RuntimeError(
            f'{mode}: class rows={len(long_frame)}, summary rows={len(summary)}'
        )
    sums = long_frame.groupby('image_id').pixel_ratio.sum().to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-8):
        raise RuntimeError(f'{mode}: pixel ratios do not sum to one')
    return {
        'seconds': float(summary.inference_seconds.sum()),
        'peak_mib': float(summary.peak_gpu_memory_mib.max()),
        'mean_classes': float(summary.classes_present.mean()),
    }


def main() -> None:
    args = parse_args()
    audit = args.audit_dir.resolve()
    manifest = pd.read_csv(audit / 'audit_manifest.csv')
    config = json.loads((audit / 'run_config.json').read_text(encoding='utf-8'))
    class_count = len(config['classes'])
    image_count = len(manifest)
    if image_count != 30:
        raise RuntimeError(f'Expected 30 audit images, got {image_count}')

    baseline = validate_mode(audit, 'baseline', image_count, class_count)
    sliding = validate_mode(audit, 'sliding_512_o25', image_count, class_count)
    comparison = pd.read_csv(audit / 'baseline_vs_sliding.csv')
    if len(comparison) != image_count:
        raise RuntimeError(f'Expected {image_count} comparison rows, got {len(comparison)}')

    city_counts = manifest.groupby(['city', 'vp_stratum']).size()
    if len(city_counts) != 15 or not (city_counts == 2).all():
        raise RuntimeError('Audit sample is not two images per city/VP stratum')

    report = f"""# DeepLabV3+ Mapillary Vistas v2.0 目标域审计

## 状态

Mapillary 训练后模型的 30 图分层审计和两种推理实验已经完成。人工评分仍需
在 `visual_review/index.html` 中完成。

## 模型

- 配置：`{config['config']}`
- checkpoint：`{config['checkpoint']}`
- 类别数：{class_count}
- 审计图像：{image_count}（每城市高/中/低 VP 各 2 张）

## 完整性检查

- baseline：{image_count} 个原尺寸 mask、{image_count} 个 overlay、
  {image_count * class_count} 行类别统计。
- sliding：{image_count} 个原尺寸 mask、{image_count} 个 overlay、
  {image_count * class_count} 行类别统计。
- 每张图的类别像素比例之和均为 1；不存在缺失输出。
- baseline 总推理时间 {baseline['seconds']:.2f} 秒，GPU 峰值
  {baseline['peak_mib']:.1f} MiB。
- sliding 总推理时间 {sliding['seconds']:.2f} 秒，GPU 峰值
  {sliding['peak_mib']:.1f} MiB。

## 方法敏感性

- 平均像素一致率：{comparison.pixel_agreement.mean():.4f}
- 中位像素一致率：{comparison.pixel_agreement.median():.4f}
- 一致率范围：{comparison.pixel_agreement.min():.4f}--{comparison.pixel_agreement.max():.4f}
- 方法间平均类别 IoU：{comparison.mean_iou_between_methods.mean():.4f}
- 平均检测类别数：baseline {baseline['mean_classes']:.2f}，sliding
  {sliding['mean_classes']:.2f}

以上比较衡量两种推理策略的一致程度，不代表真实准确率。最终方法必须结合
人工审核决定。

## 人工审核

打开 `visual_review/index.html`，评分标准：0=不可用或严重错误，1=部分可用，
2=总体可靠，NA=无法判断。重点检查广告牌、店招、道路、人行道、杆件、路灯、
垃圾桶、坑洞、临时护栏、车辆和植被等 Mapillary 类别，并记录滑窗接缝、
小目标遗漏和类别混淆。
"""
    (audit / 'REPORT.md').write_text(report, encoding='utf-8')
    print(f'Validated audit and wrote {audit / "REPORT.md"}')


if __name__ == '__main__':
    main()

