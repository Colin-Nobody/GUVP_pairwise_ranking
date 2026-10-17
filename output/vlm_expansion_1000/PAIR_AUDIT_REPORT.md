# 正式 MEC+BPC 审计摘要

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 提示词版本：`vp-faireval-mec-bpc-score-v6`
- MEC：每个位置 k=3，temperature=1.0
- 已完成无序图片对：6500
- MEC 方向采样次数：39000
- 有效校准票：5146（79.2%）
- 校准平局：270
- 无效对：1084
- 平均绝对校准分差：1.1082
- 平均 BPDE：0.4942
- 第一位置平均分优势：0.5534

## 状态计数

- `calibrated`：5146
- `calibrated_tie`：270
- `invalid_ab`：490
- `invalid_ba`：528
- `invalid_both`：66
