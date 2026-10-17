# 正式 MEC+BPC 审计摘要

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 提示词版本：`vp-faireval-mec-bpc-score-v6`
- MEC：每个位置 k=3，temperature=1.0
- 已完成无序图片对：3000
- MEC 方向采样次数：18000
- 有效校准票：2346（78.2%）
- 校准平局：138
- 无效对：516
- 平均绝对校准分差：0.9858
- 平均 BPDE：0.5364
- 第一位置平均分优势：0.5761

## 状态计数

- `calibrated`：2346
- `calibrated_tie`：138
- `invalid_ab`：240
- `invalid_ba`：249
- `invalid_both`：27
